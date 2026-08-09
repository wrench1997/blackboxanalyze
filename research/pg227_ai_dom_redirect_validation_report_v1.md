# PG-227 AI DOM/redirect validation

fresh=14; routes=7; AI=14; reference=14; negative=14
DOM surface effect=4; redirect effect=0; xss_positive=0; open_redirect_positive=0; false_positive=0

DOM marker effect is not XSS: JavaScript was disabled and browser network access was aborted. A normal same-origin redirect is not an open-redirect positive. Wire values remain runtime placeholders.

- GET /vul/xss/xss_01.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=True; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_01.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_02.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_02.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_03.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_03.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_04.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_04.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_reflected_get.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=True; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_reflected_get.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_dom_x.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_dom_x.php?text=<RUNTIME_CANARY>`
- GET /vul/urlredirect/urlredirect.php: probe=http_canary; modality=typed_redirect_shape; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/urlredirect/urlredirect.php?url=<RUNTIME_CANARY>`
- GET /vul/xss/xss_01.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=True; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_01.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_02.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_02.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_03.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_03.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_04.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_04.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_reflected_get.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=True; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_reflected_get.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_dom_x.php: probe=inert_dom_markup; modality=typed_dom_surface_effect; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_dom_x.php?text=<RUNTIME_CANARY>`
- GET /vul/urlredirect/urlredirect.php: probe=http_canary; modality=typed_redirect_shape; dom_effect=False; redirect_effect=False; wire=`GET <LOOPBACK_ORIGIN>/vul/urlredirect/urlredirect.php?url=<RUNTIME_CANARY>`
