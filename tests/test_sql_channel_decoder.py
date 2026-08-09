import torch

from app.sql_channel_decoder import SQL_CHANNEL_FEATURE_DIM, SqlChannelDecoder, sql_channel_feature_vector


def test_sql_channel_features_exclude_oracle_projection():
    row = {
        "payload": {"probe_kind": "sql_channel_class", "probe": "syntax_error"},
        "response_projection": {"status_code": 422, "body_length": 48, "headers": {"content-type": "application/json"}, "json_shape": {"key_count": 3, "type": "object"}},
        "oracle_projection": {"interpreter_boundary": True, "candidate_ast_sha256": "secret"},
    }
    vector = sql_channel_feature_vector(row)
    assert len(vector) == SQL_CHANNEL_FEATURE_DIM
    assert vector[2] == 1.0
    assert "secret" not in str(vector)


def test_sql_channel_decoder_can_abstain():
    model = SqlChannelDecoder()
    output = model.decode(torch.zeros(2, SQL_CHANNEL_FEATURE_DIM), abstain_threshold=0.0)
    assert len(output) == 2
    assert output[0]["abstained"] is False
