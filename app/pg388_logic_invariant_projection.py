"""PG-388 abstract business-logic/invariant reasoning projection.

The module describes *why* a local fixture accepted or rejected a transition.
It does not contain a route, account, credential, payload, response body, or
wire value.  The model-facing side is a causal sequence of abstract state and
invariant tokens; a separate local evaluator may later bind a bounded canary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "pg388-logic-invariant-projection-v1"
ROLES = ("candidate", "reference", "negative", "replay")
FEEDBACK_STATES = ("baseline", "missing", "invariant_mismatch", "state_mismatch", "typed_effect")

LOGIC_CASES: tuple[dict[str, str], ...] = (
    {"case_ref": "subject_resource_scope", "surface": "authorization_boundary", "state_model": "stateless", "invariant": "subject_matches_resource", "precondition": "authenticated_subject", "transition": "read_protected_resource", "counterfactual": "different_subject_same_resource", "observation": "access_decision_shape", "failure": "scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "protected_resource_shape"},
    {"case_ref": "tenant_scope_boundary", "surface": "tenant_isolation", "state_model": "stateless", "invariant": "tenant_matches_resource", "precondition": "tenant_context_present", "transition": "read_tenant_record", "counterfactual": "cross_tenant_context", "observation": "record_shape", "failure": "tenant_scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "tenant_record_shape"},
    {"case_ref": "role_transition_precondition", "surface": "role_transition", "state_model": "history_dependent", "invariant": "approval_precedes_transition", "precondition": "approval_state_present", "transition": "role_change", "counterfactual": "transition_without_approval", "observation": "state_transition_shape", "failure": "precondition_missing", "default_action": "ask", "repair": "precondition", "oracle": "role_state_shape"},
    {"case_ref": "quota_boundary", "surface": "business_boundary", "state_model": "stateless", "invariant": "requested_value_within_quota", "precondition": "quota_present", "transition": "issue_benefit", "counterfactual": "boundary_plus_one", "observation": "issued_value_shape", "failure": "boundary_mismatch", "default_action": "repair", "repair": "boundary", "oracle": "benefit_state_shape"},
    {"case_ref": "coupon_reuse_idempotency", "surface": "idempotency", "state_model": "history_dependent", "invariant": "coupon_consumed_once", "precondition": "coupon_unused", "transition": "apply_benefit", "counterfactual": "same_action_replay", "observation": "duplicate_transition_shape", "failure": "replay_accepted", "default_action": "repair", "repair": "replay", "oracle": "idempotency_state_shape"},
    {"case_ref": "workflow_order", "surface": "workflow_state_machine", "state_model": "history_dependent", "invariant": "state_transition_ordered", "precondition": "prior_step_complete", "transition": "advance_workflow", "counterfactual": "skip_prior_step", "observation": "workflow_state_shape", "failure": "order_mismatch", "default_action": "repair", "repair": "sequence", "oracle": "workflow_state_shape"},
    {"case_ref": "nonce_replay", "surface": "replay_protection", "state_model": "history_dependent", "invariant": "nonce_fresh_per_action", "precondition": "fresh_nonce", "transition": "accept_action", "counterfactual": "reused_nonce", "observation": "replay_decision_shape", "failure": "replay_accepted", "default_action": "repair", "repair": "replay", "oracle": "nonce_decision_shape"},
    {"case_ref": "object_state_machine", "surface": "object_transition", "state_model": "history_dependent", "invariant": "transition_allowed_from_current_state", "precondition": "current_state_known", "transition": "change_object_state", "counterfactual": "stale_state_transition", "observation": "state_delta_shape", "failure": "state_mismatch", "default_action": "ask", "repair": "state", "oracle": "object_state_shape"},
    {"case_ref": "separation_of_duties", "surface": "approval_boundary", "state_model": "history_dependent", "invariant": "requester_differs_from_approver", "precondition": "independent_approver", "transition": "finalize_approval", "counterfactual": "same_subject_approves", "observation": "approval_state_shape", "failure": "role_overlap", "default_action": "ask", "repair": "role", "oracle": "approval_state_shape"},
    {"case_ref": "redirect_state_binding", "surface": "redirect_contract", "state_model": "history_dependent", "invariant": "redirect_state_binds_to_session", "precondition": "state_token_present", "transition": "complete_redirect", "counterfactual": "state_from_other_session", "observation": "redirect_state_shape", "failure": "state_binding_mismatch", "default_action": "ask", "repair": "state", "oracle": "redirect_state_shape"},
    {"case_ref": "truthiness_boundary", "surface": "type_coercion", "state_model": "stateless", "invariant": "typed_value_matches_policy", "precondition": "value_type_observed", "transition": "authorize_boolean_branch", "counterfactual": "empty_or_zero_coercion", "observation": "boolean_branch_shape", "failure": "coercion_mismatch", "default_action": "ask", "repair": "type", "oracle": "boolean_branch_shape"},
    {"case_ref": "client_server_validation", "surface": "validation_consistency", "state_model": "stateless", "invariant": "client_and_server_constraints_agree", "precondition": "validation_contract_observed", "transition": "accept_business_action", "counterfactual": "client_only_validation", "observation": "validation_result_shape", "failure": "validation_divergence", "default_action": "ask", "repair": "validation", "oracle": "validation_result_shape"},
)

# The core cases above are the compact demo.  This append-only inventory maps
# the common logic-vulnerability taxonomy to the same abstract contract.  It
# intentionally records invariants and state transitions rather than steps,
# identifiers, credentials or exploit values.
LOGIC_CASES = LOGIC_CASES + (
    {"case_ref": "install_reentry_gate", "surface": "installation_gate", "state_model": "history_dependent", "invariant": "install_once_until_reset", "precondition": "installation_state_known", "transition": "complete_install", "counterfactual": "reenter_install", "observation": "install_state_shape", "failure": "reentry_accepted", "default_action": "repair", "repair": "state", "oracle": "install_state_shape"},
    {"case_ref": "install_artifact_exposure", "surface": "installation_artifact", "state_model": "stateless", "invariant": "setup_artifact_not_exposed", "precondition": "deployment_state_known", "transition": "serve_static_artifact", "counterfactual": "artifact_after_install", "observation": "artifact_access_shape", "failure": "artifact_exposed", "default_action": "ask", "repair": "scope", "oracle": "artifact_access_shape"},
    {"case_ref": "update_authorization_gate", "surface": "update_boundary", "state_model": "history_dependent", "invariant": "update_requires_owner_scope", "precondition": "owner_scope_present", "transition": "apply_update", "counterfactual": "update_other_scope", "observation": "update_state_shape", "failure": "scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "update_state_shape"},
    {"case_ref": "purchase_price_binding", "surface": "transaction_price", "state_model": "stateless", "invariant": "price_bound_to_server_quote", "precondition": "server_quote_present", "transition": "create_order", "counterfactual": "client_price_differs", "observation": "order_total_shape", "failure": "price_divergence", "default_action": "ask", "repair": "binding", "oracle": "order_total_shape"},
    {"case_ref": "purchase_status_transition", "surface": "transaction_status", "state_model": "history_dependent", "invariant": "status_transition_is_server_owned", "precondition": "order_state_known", "transition": "confirm_payment", "counterfactual": "client_status_claim", "observation": "order_status_shape", "failure": "status_mismatch", "default_action": "ask", "repair": "state", "oracle": "order_status_shape"},
    {"case_ref": "purchase_quantity_floor", "surface": "transaction_quantity", "state_model": "stateless", "invariant": "quantity_is_positive", "precondition": "quantity_type_observed", "transition": "calculate_total", "counterfactual": "non_positive_quantity", "observation": "total_shape", "failure": "boundary_mismatch", "default_action": "repair", "repair": "boundary", "oracle": "total_shape"},
    {"case_ref": "purchase_amount_floor", "surface": "transaction_amount", "state_model": "stateless", "invariant": "amount_is_non_negative", "precondition": "amount_type_observed", "transition": "settle_order", "counterfactual": "negative_amount", "observation": "settlement_shape", "failure": "boundary_mismatch", "default_action": "repair", "repair": "boundary", "oracle": "settlement_shape"},
    {"case_ref": "purchase_concurrency_lock", "surface": "transaction_concurrency", "state_model": "history_dependent", "invariant": "single_commit_per_order", "precondition": "order_version_present", "transition": "commit_order", "counterfactual": "parallel_commit", "observation": "commit_version_shape", "failure": "duplicate_transition", "default_action": "ask", "repair": "replay", "oracle": "commit_version_shape"},
    {"case_ref": "coupon_reuse_boundary", "surface": "risk_coupon", "state_model": "history_dependent", "invariant": "coupon_consumed_once", "precondition": "coupon_state_present", "transition": "apply_coupon", "counterfactual": "reuse_consumed_coupon", "observation": "benefit_delta_shape", "failure": "replay_accepted", "default_action": "repair", "repair": "replay", "oracle": "benefit_delta_shape"},
    {"case_ref": "cashout_balance_binding", "surface": "risk_cashout", "state_model": "history_dependent", "invariant": "cashout_within_balance", "precondition": "balance_snapshot_present", "transition": "create_cashout", "counterfactual": "stale_balance", "observation": "balance_delta_shape", "failure": "balance_mismatch", "default_action": "ask", "repair": "state", "oracle": "balance_delta_shape"},
    {"case_ref": "registration_overwrite", "surface": "registration_identity", "state_model": "stateless", "invariant": "registration_does_not_overwrite", "precondition": "identity_canonicalization_observed", "transition": "create_identity", "counterfactual": "canonical_name_collision", "observation": "registration_result_shape", "failure": "overwrite_or_collision", "default_action": "ask", "repair": "canonicalization", "oracle": "registration_result_shape"},
    {"case_ref": "registration_enumeration", "surface": "registration_identity", "state_model": "stateless", "invariant": "existence_signal_is_uniform", "precondition": "response_shape_observed", "transition": "validate_identity", "counterfactual": "existing_vs_new_identity", "observation": "uniform_response_shape", "failure": "existence_signal", "default_action": "ask", "repair": "shape", "oracle": "uniform_response_shape"},
    {"case_ref": "password_storage_contract", "surface": "credential_storage", "state_model": "stateless", "invariant": "secret_storage_is_one_way", "precondition": "storage_projection_observed", "transition": "persist_secret", "counterfactual": "reversible_storage_shape", "observation": "storage_shape", "failure": "storage_contract_mismatch", "default_action": "ask", "repair": "storage", "oracle": "storage_shape"},
    {"case_ref": "password_strength_contract", "surface": "credential_policy", "state_model": "stateless", "invariant": "credential_policy_enforced_server_side", "precondition": "policy_shape_observed", "transition": "set_secret", "counterfactual": "weak_policy_input", "observation": "policy_result_shape", "failure": "policy_divergence", "default_action": "ask", "repair": "policy", "oracle": "policy_result_shape"},
    {"case_ref": "identity_canonicalization", "surface": "identity_canonicalization", "state_model": "stateless", "invariant": "canonical_identity_is_unique", "precondition": "normalization_order_observed", "transition": "resolve_identity", "counterfactual": "case_or_space_variant", "observation": "identity_resolution_shape", "failure": "canonicalization_mismatch", "default_action": "repair", "repair": "canonicalization", "oracle": "identity_resolution_shape"},
    {"case_ref": "cookie_integrity", "surface": "session_cookie", "state_model": "history_dependent", "invariant": "client_state_is_integrity_bound", "precondition": "integrity_metadata_present", "transition": "accept_session_state", "counterfactual": "modified_client_state", "observation": "session_decision_shape", "failure": "integrity_mismatch", "default_action": "ask", "repair": "integrity", "oracle": "session_decision_shape"},
    {"case_ref": "phone_canonicalization", "surface": "phone_identity", "state_model": "stateless", "invariant": "phone_forms_canonicalize_equally", "precondition": "phone_normalization_observed", "transition": "resolve_phone_identity", "counterfactual": "country_prefix_variant", "observation": "identity_resolution_shape", "failure": "canonicalization_mismatch", "default_action": "repair", "repair": "canonicalization", "oracle": "identity_resolution_shape"},
    {"case_ref": "login_rate_limit", "surface": "login_control", "state_model": "history_dependent", "invariant": "attempt_budget_decreases", "precondition": "attempt_counter_present", "transition": "authenticate", "counterfactual": "repeated_failure_sequence", "observation": "lock_state_shape", "failure": "rate_limit_mismatch", "default_action": "ask", "repair": "sequence", "oracle": "lock_state_shape"},
    {"case_ref": "lock_unlock_contract", "surface": "login_lock_state", "state_model": "history_dependent", "invariant": "lock_has_reviewed_unlock_path", "precondition": "lock_state_known", "transition": "unlock_account", "counterfactual": "unlock_without_recovery", "observation": "unlock_state_shape", "failure": "unlock_mismatch", "default_action": "ask", "repair": "state", "oracle": "unlock_state_shape"},
    {"case_ref": "session_storage_boundary", "surface": "session_storage", "state_model": "history_dependent", "invariant": "session_secret_not_client_visible", "precondition": "storage_projection_observed", "transition": "restore_session", "counterfactual": "client_storage_visible", "observation": "session_storage_shape", "failure": "storage_exposure", "default_action": "ask", "repair": "storage", "oracle": "session_storage_shape"},
    {"case_ref": "password_reset_subject_binding", "surface": "password_reset", "state_model": "history_dependent", "invariant": "reset_subject_matches_requester", "precondition": "reset_state_present", "transition": "complete_reset", "counterfactual": "different_subject_reference", "observation": "reset_result_shape", "failure": "subject_binding_mismatch", "default_action": "ask", "repair": "scope", "oracle": "reset_result_shape"},
    {"case_ref": "password_reset_expiry", "surface": "password_reset", "state_model": "history_dependent", "invariant": "reset_state_expires", "precondition": "expiry_observed", "transition": "accept_reset", "counterfactual": "expired_reset_state", "observation": "expiry_decision_shape", "failure": "expiry_mismatch", "default_action": "ask", "repair": "state", "oracle": "expiry_decision_shape"},
    {"case_ref": "password_reset_response_secrecy", "surface": "password_reset", "state_model": "stateless", "invariant": "new_secret_not_in_response", "precondition": "response_projection_observed", "transition": "return_reset_result", "counterfactual": "secret_echo_shape", "observation": "response_shape", "failure": "secret_exposure", "default_action": "ask", "repair": "shape", "oracle": "response_shape"},
    {"case_ref": "password_reset_host_binding", "surface": "password_reset_host", "state_model": "history_dependent", "invariant": "reset_origin_is_allowlisted", "precondition": "origin_policy_observed", "transition": "build_reset_link", "counterfactual": "untrusted_origin", "observation": "origin_shape", "failure": "origin_policy_mismatch", "default_action": "ask", "repair": "scope", "oracle": "origin_shape"},
    {"case_ref": "change_password_old_secret", "surface": "password_change", "state_model": "history_dependent", "invariant": "old_secret_required", "precondition": "authenticated_subject", "transition": "change_secret", "counterfactual": "missing_old_secret", "observation": "change_result_shape", "failure": "precondition_missing", "default_action": "ask", "repair": "precondition", "oracle": "change_result_shape"},
    {"case_ref": "appeal_identity_binding", "surface": "appeal_workflow", "state_model": "history_dependent", "invariant": "appeal_subject_is_verified", "precondition": "identity_evidence_present", "transition": "review_appeal", "counterfactual": "unverified_subject", "observation": "appeal_state_shape", "failure": "identity_mismatch", "default_action": "ask", "repair": "identity", "oracle": "appeal_state_shape"},
    {"case_ref": "update_field_allowlist", "surface": "update_contract", "state_model": "stateless", "invariant": "only_declared_fields_update", "precondition": "field_allowlist_observed", "transition": "apply_update", "counterfactual": "undeclared_field", "observation": "update_field_shape", "failure": "field_scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "update_field_shape"},
    {"case_ref": "query_object_scope", "surface": "query_authorization", "state_model": "stateless", "invariant": "query_subject_owns_object", "precondition": "ownership_scope_present", "transition": "read_object", "counterfactual": "different_object_owner", "observation": "object_visibility_shape", "failure": "horizontal_scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "object_visibility_shape"},
    {"case_ref": "query_identifier_entropy", "surface": "identifier_enumeration", "state_model": "stateless", "invariant": "identifier_not_predictable", "precondition": "identifier_shape_observed", "transition": "resolve_object", "counterfactual": "neighbor_identifier", "observation": "lookup_shape", "failure": "enumeration_signal", "default_action": "ask", "repair": "shape", "oracle": "lookup_shape"},
    {"case_ref": "two_factor_reset_binding", "surface": "two_factor_reset", "state_model": "history_dependent", "invariant": "reset_does_not_skip_second_factor", "precondition": "second_factor_state_present", "transition": "restore_account", "counterfactual": "reset_then_auto_login", "observation": "auth_state_shape", "failure": "factor_bypass", "default_action": "ask", "repair": "state", "oracle": "auth_state_shape"},
    {"case_ref": "two_factor_attempt_budget", "surface": "two_factor_attempts", "state_model": "history_dependent", "invariant": "factor_attempts_are_bounded", "precondition": "attempt_counter_present", "transition": "verify_factor", "counterfactual": "repeated_factor_failure", "observation": "factor_lock_shape", "failure": "attempt_budget_mismatch", "default_action": "ask", "repair": "sequence", "oracle": "factor_lock_shape"},
    {"case_ref": "captcha_reuse", "surface": "captcha_state", "state_model": "history_dependent", "invariant": "challenge_consumed_once", "precondition": "challenge_state_present", "transition": "accept_verification", "counterfactual": "reuse_consumed_challenge", "observation": "verification_state_shape", "failure": "replay_accepted", "default_action": "repair", "repair": "replay", "oracle": "verification_state_shape"},
    {"case_ref": "captcha_expiry", "surface": "captcha_expiry", "state_model": "history_dependent", "invariant": "challenge_expires", "precondition": "expiry_state_present", "transition": "accept_verification", "counterfactual": "expired_challenge", "observation": "expiry_shape", "failure": "expiry_mismatch", "default_action": "ask", "repair": "state", "oracle": "expiry_shape"},
    {"case_ref": "captcha_attempt_limit", "surface": "captcha_attempts", "state_model": "history_dependent", "invariant": "challenge_attempts_bounded", "precondition": "attempt_counter_present", "transition": "verify_challenge", "counterfactual": "many_attempts", "observation": "attempt_budget_shape", "failure": "attempt_budget_mismatch", "default_action": "ask", "repair": "sequence", "oracle": "attempt_budget_shape"},
    {"case_ref": "session_fixation_boundary", "surface": "session_rotation", "state_model": "history_dependent", "invariant": "session_rotates_after_auth", "precondition": "pre_auth_session_present", "transition": "complete_authentication", "counterfactual": "reuse_pre_auth_session", "observation": "session_identity_shape", "failure": "session_reuse", "default_action": "ask", "repair": "state", "oracle": "session_identity_shape"},
    {"case_ref": "session_scope_boundary", "surface": "session_scope", "state_model": "history_dependent", "invariant": "session_scope_matches_subject", "precondition": "session_subject_present", "transition": "restore_session", "counterfactual": "different_subject_session", "observation": "session_decision_shape", "failure": "subject_binding_mismatch", "default_action": "ask", "repair": "scope", "oracle": "session_decision_shape"},
    {"case_ref": "unauthorized_static_resource", "surface": "static_resource_access", "state_model": "stateless", "invariant": "protected_resource_requires_scope", "precondition": "resource_policy_observed", "transition": "serve_resource", "counterfactual": "anonymous_request", "observation": "resource_access_shape", "failure": "authorization_missing", "default_action": "ask", "repair": "scope", "oracle": "resource_access_shape"},
    {"case_ref": "vertical_role_scope", "surface": "vertical_authorization", "state_model": "stateless", "invariant": "role_allows_transition", "precondition": "role_context_present", "transition": "invoke_privileged_action", "counterfactual": "lower_role_context", "observation": "role_decision_shape", "failure": "vertical_scope_mismatch", "default_action": "ask", "repair": "role", "oracle": "role_decision_shape"},
    {"case_ref": "cross_role_scope", "surface": "cross_authorization", "state_model": "stateless", "invariant": "subject_and_role_both_match", "precondition": "subject_role_context_present", "transition": "invoke_scoped_action", "counterfactual": "different_subject_and_role", "observation": "role_scope_shape", "failure": "cross_scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "role_scope_shape"},
    {"case_ref": "random_seed_quality", "surface": "randomness_contract", "state_model": "stateless", "invariant": "random_state_not_predictable", "precondition": "randomness_source_observed", "transition": "issue_state_token", "counterfactual": "repeated_seed_shape", "observation": "randomness_shape", "failure": "predictability_signal", "default_action": "ask", "repair": "entropy", "oracle": "randomness_shape"},
    {"case_ref": "rate_limit_scope", "surface": "rate_limit_contract", "state_model": "history_dependent", "invariant": "attempt_budget_binds_to_subject_and_action", "precondition": "rate_limit_key_present", "transition": "accept_action", "counterfactual": "key_scope_variant", "observation": "rate_limit_shape", "failure": "limit_scope_mismatch", "default_action": "ask", "repair": "scope", "oracle": "rate_limit_shape"},
    {"case_ref": "crypto_mode_binding", "surface": "crypto_contract", "state_model": "stateless", "invariant": "crypto_parameters_are_bound", "precondition": "algorithm_contract_observed", "transition": "verify_record", "counterfactual": "parameter_variant", "observation": "verification_shape", "failure": "crypto_contract_mismatch", "default_action": "ask", "repair": "binding", "oracle": "verification_shape"},
    {"case_ref": "execution_order", "surface": "execution_order", "state_model": "history_dependent", "invariant": "checks_precede_side_effect", "precondition": "ordered_steps_observed", "transition": "commit_action", "counterfactual": "side_effect_before_check", "observation": "order_trace_shape", "failure": "order_mismatch", "default_action": "ask", "repair": "sequence", "oracle": "order_trace_shape"},
    {"case_ref": "sensitive_projection", "surface": "information_projection", "state_model": "stateless", "invariant": "sensitive_fields_not_projected", "precondition": "response_projection_observed", "transition": "return_summary", "counterfactual": "field_projection_variant", "observation": "field_shape", "failure": "sensitive_field_exposed", "default_action": "ask", "repair": "shape", "oracle": "field_shape"},
)

# Supplemental contracts for the fine-grained gaps identified by the 4.13
# taxonomy audit.  They are append-only and remain separate from the frozen
# v1 dataset until their own dataset and replay audit pass.
SUPPLEMENTAL_LOGIC_CASES: tuple[dict[str, str], ...] = (
    {"case_ref": "oauth_second_factor", "surface": "oauth_authentication", "state_model": "history_dependent", "invariant": "oauth_session_requires_second_factor", "precondition": "oauth_session_pending", "transition": "issue_session", "counterfactual": "oauth_without_factor", "observation": "auth_upgrade_shape", "failure": "factor_bypass", "default_action": "ask", "repair": "precondition", "oracle": "auth_upgrade_shape"},
    {"case_ref": "activation_link_second_factor", "surface": "activation_flow", "state_model": "history_dependent", "invariant": "activation_does_not_skip_factor", "precondition": "activation_state_pending", "transition": "activate_account", "counterfactual": "activation_without_factor", "observation": "activation_state_shape", "failure": "factor_bypass", "default_action": "ask", "repair": "state", "oracle": "activation_state_shape"},
    {"case_ref": "csrf_disable_second_factor", "surface": "factor_settings", "state_model": "history_dependent", "invariant": "factor_disable_requires_csrf_and_reauth", "precondition": "factor_enabled", "transition": "change_factor_setting", "counterfactual": "cross_origin_disable", "observation": "factor_setting_shape", "failure": "csrf_binding_missing", "default_action": "ask", "repair": "binding", "oracle": "factor_setting_shape"},
    {"case_ref": "captcha_predictability", "surface": "captcha_entropy", "state_model": "stateless", "invariant": "challenge_value_not_predictable", "precondition": "challenge_shape_observed", "transition": "issue_challenge", "counterfactual": "neighbor_challenge", "observation": "challenge_entropy_bucket", "failure": "predictability_signal", "default_action": "ask", "repair": "entropy", "oracle": "challenge_entropy_bucket"},
    {"case_ref": "captcha_response_exposure", "surface": "captcha_response", "state_model": "stateless", "invariant": "challenge_value_not_returned", "precondition": "response_projection_observed", "transition": "return_challenge_result", "counterfactual": "challenge_echo_shape", "observation": "redacted_response_shape", "failure": "challenge_exposed", "default_action": "ask", "repair": "shape", "oracle": "redacted_response_shape"},
    {"case_ref": "captcha_client_validation", "surface": "captcha_validation", "state_model": "stateless", "invariant": "challenge_validation_is_server_owned", "precondition": "server_validation_observed", "transition": "verify_challenge", "counterfactual": "client_only_validation", "observation": "validation_result_shape", "failure": "client_server_divergence", "default_action": "ask", "repair": "validation", "oracle": "validation_result_shape"},
    {"case_ref": "captcha_delivery_abuse", "surface": "captcha_delivery", "state_model": "history_dependent", "invariant": "challenge_delivery_is_rate_limited", "precondition": "delivery_budget_present", "transition": "issue_challenge", "counterfactual": "repeated_delivery", "observation": "delivery_budget_shape", "failure": "delivery_limit_missing", "default_action": "ask", "repair": "scope", "oracle": "delivery_budget_shape"},
    {"case_ref": "session_guessing", "surface": "session_entropy", "state_model": "stateless", "invariant": "session_identifier_not_predictable", "precondition": "identifier_projection_observed", "transition": "issue_session", "counterfactual": "neighbor_session", "observation": "session_entropy_bucket", "failure": "predictability_signal", "default_action": "ask", "repair": "entropy", "oracle": "session_entropy_bucket"},
    {"case_ref": "session_forgery", "surface": "session_integrity", "state_model": "history_dependent", "invariant": "session_state_integrity_bound", "precondition": "integrity_contract_observed", "transition": "restore_session", "counterfactual": "modified_session_state", "observation": "session_decision_shape", "failure": "integrity_mismatch", "default_action": "ask", "repair": "binding", "oracle": "session_decision_shape"},
    {"case_ref": "session_leakage", "surface": "session_exposure", "state_model": "history_dependent", "invariant": "session_secret_not_projected", "precondition": "storage_projection_observed", "transition": "restore_session", "counterfactual": "client_visible_session", "observation": "session_storage_shape", "failure": "storage_exposure", "default_action": "ask", "repair": "storage", "oracle": "session_storage_shape"},
)

ALL_LOGIC_CASES: tuple[dict[str, str], ...] = LOGIC_CASES + SUPPLEMENTAL_LOGIC_CASES

# Keep ontology words such as ``credential_storage`` and ``wire_shape`` usable
# as abstract labels.  Reject only literal/value-shaped fields or executable
# material, never the category name itself.
_FORBIDDEN = ("http://", "https://", "payload=", "payload:", "wire=", "wire:", "response_body=", "response_body:", "cookie_value=", "cookie_value:", "credential=", "credential:", "callback=", "callback:", "webhook=", "webhook:", "<script")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _case(case_ref: str) -> dict[str, str]:
    for item in ALL_LOGIC_CASES:
        if item["case_ref"] == case_ref:
            return dict(item)
    raise ValueError("unknown_pg388_logic_case")


def _reject_raw(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_reject_raw(key) or _reject_raw(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_reject_raw(item) for item in value)
    return isinstance(value, str) and any(fragment in value.casefold() for fragment in _FORBIDDEN)


def _context_tokens(case: Mapping[str, str], *, role: str, feedback_state: str) -> list[str]:
    ordered = (
        ("logic_surface", case["surface"]),
        ("state_model", case["state_model"]),
        ("logic_invariant", case["invariant"]),
        ("precondition", case["precondition"]),
        ("transition", case["transition"]),
        ("counterfactual", case["counterfactual"]),
        ("observation_shape", case["observation"]),
        ("failure_shape", case["failure"] if feedback_state != "baseline" else "none"),
        ("feedback_state", feedback_state),
        ("history_action", "baseline_observe" if feedback_state in {"baseline", "missing"} else "select_probe_variant"),
        ("role", role),
        ("evidence_scope", "matched_triplet" if role == "negative" else "required"),
    )
    return ["[CTX_BOS]", *(f"{key}={value}" for key, value in ordered), "context_firewall=closed", "sidecars_off_context=true", "[CTX_EOS]"]


def project_logic_case(case_ref: str, *, role: str = "candidate", feedback_state: str = "invariant_mismatch") -> dict[str, Any]:
    """Project one business-logic case to context and Rule-IR target tokens."""

    case = _case(str(case_ref))
    role = str(role)
    feedback_state = str(feedback_state)
    if role not in ROLES:
        raise ValueError("unknown_pg388_role")
    if feedback_state not in FEEDBACK_STATES:
        raise ValueError("unknown_pg388_feedback_state")
    question = "none"
    next_action = case["default_action"]
    repair = case["repair"]
    variant = "one_variable_logic_repair"
    ask_reason = "none"
    if role == "negative":
        next_action, repair, variant, ask_reason = "abstain", "none", "matched_negative_control", "matched_negative_control"
    elif feedback_state in {"missing", "baseline"} or case["default_action"] == "ask":
        question, next_action, repair, variant, ask_reason = "ask_logic_observation", "ask", "observe", "unsupported_abstain", "missing_state_or_invariant_observation"
    elif feedback_state == "typed_effect":
        next_action, repair, variant = "replay", "replay", "fresh_replay"
    elif feedback_state in {"invariant_mismatch", "state_mismatch"}:
        next_action, variant = "repair", "one_variable_logic_repair"
    target = {
        "question": question,
        "ask_reason": ask_reason,
        "next_action": next_action,
        "repair_action": repair,
        "logic_invariant_ref": case["invariant"],
        "state_transition_ref": case["transition"],
        "precondition_ref": case["precondition"],
        "counterfactual_ref": case["counterfactual"],
        "probe_variant_ref": variant,
        "oracle_ref": case["oracle"] if role != "negative" else "negative_no_effect",
        "safe_to_send": False,
    }
    projection = {
        "schema_version": SCHEMA_VERSION,
        "case_ref": case["case_ref"],
        "role": role,
        "feedback_state": feedback_state,
        "context_tokens": _context_tokens(case, role=role, feedback_state=feedback_state),
        "target_tokens": ["[TARGET_BOS]", *(f"{key}={value}" for key, value in target.items()), "[TARGET_EOS]"],
        "logic_context": {key: case[key] for key in ("surface", "state_model", "invariant", "precondition", "transition", "counterfactual", "observation", "failure")},
        "target_projection": target,
        "raw_source_stored": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    projection["projection_sha256"] = _sha(projection)
    return projection


def project_logic_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an abstract observation to ASK/repair/replay Rule-IR."""

    if not isinstance(observation, Mapping) or _reject_raw(observation):
        return {"schema_version": SCHEMA_VERSION, "status": "rejected_raw_observation", "next_action": "ask", "ask_reason": "raw_or_evaluator_observation", "safe_to_send": False}
    required = ("case_ref", "feedback_state")
    missing = [key for key in required if observation.get(key) in (None, "", "unknown", "not_observed")]
    if missing:
        return {"schema_version": SCHEMA_VERSION, "status": "ask_missing_logic_observation", "next_action": "ask", "ask_reason": "missing:" + ",".join(missing), "safe_to_send": False}
    result = project_logic_case(str(observation["case_ref"]), role=str(observation.get("role", "candidate")), feedback_state=str(observation["feedback_state"]))
    result["status"] = "ask" if result["target_projection"]["next_action"] == "ask" else "abstract_logic_variant_selected"
    result["input_sha256"] = _sha(observation)
    return result


__all__ = ["ALL_LOGIC_CASES", "FEEDBACK_STATES", "LOGIC_CASES", "ROLES", "SCHEMA_VERSION", "SUPPLEMENTAL_LOGIC_CASES", "project_logic_case", "project_logic_observation"]
