#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

import json
from typing import Any, Optional

import pytest
import requests

from airbyte_cdk.sources.declarative.extractors import DpathExtractor
from airbyte_cdk.sources.declarative.requesters.paginators.strategies.offset_increment import (
    OffsetIncrement,
)


@pytest.mark.parametrize(
    "page_size, parameters, response_results, last_page_size, last_record, last_page_token_value, expected_next_page_token, expected_offset",
    [
        pytest.param(
            "2", {}, [{"id": 1}, {"id": 2}], 2, {"id": 2}, 4, 6, 2, id="test_same_page_size"
        ),
        pytest.param(
            2, {}, [{"id": 1}, {"id": 2}], 2, {"id": 2}, 4, 6, 2, id="test_same_page_size"
        ),
        pytest.param(
            "{{ parameters['page_size'] }}",
            {"page_size": 3},
            [{"id": 1}, {"id": 2}],
            2,
            {"id": 1},
            3,
            None,
            0,
            id="test_larger_page_size",
        ),
        pytest.param(None, {}, [], 0, [], 3, None, 0, id="test_stop_if_no_records"),
        pytest.param(
            "{{ response['page_metadata']['limit'] }}",
            {},
            [{"id": 1}, {"id": 2}],
            2,
            {"id": 2},
            3,
            None,
            0,
            id="test_page_size_from_response",
        ),
        pytest.param(
            2,
            {},
            [{"id": 1}, {"id": 2}],
            2,
            {"id": 2},
            None,
            2,
            2,
            id="test_get_second_page_with_first_page_not_injected",
        ),
    ],
)
def test_offset_increment_paginator_strategy(
    page_size,
    parameters,
    response_results,
    last_page_size,
    last_record,
    last_page_token_value,
    expected_next_page_token,
    expected_offset,
):
    extractor = DpathExtractor(field_path=["results"], parameters={}, config={})
    paginator_strategy = OffsetIncrement(
        page_size=page_size, extractor=extractor, parameters=parameters, config={}
    )

    response = requests.Response()

    response.headers = {"A_HEADER": "HEADER_VALUE"}
    response_body = {
        "results": response_results,
        "next": "https://airbyte.io/next_url",
        "page_metadata": {"limit": 5},
    }
    response._content = json.dumps(response_body).encode("utf-8")

    next_page_token = paginator_strategy.next_page_token(
        response, last_page_size, last_record, last_page_token_value
    )
    assert expected_next_page_token == next_page_token

    # Validate that the PaginationStrategy is stateless and calling next_page_token() again returns the same result
    next_page_token = paginator_strategy.next_page_token(
        response, last_page_size, last_record, last_page_token_value
    )
    assert expected_next_page_token == next_page_token


def test_offset_increment_response_without_record_path():
    extractor = DpathExtractor(field_path=["results"], parameters={}, config={})
    paginator_strategy = OffsetIncrement(page_size=2, extractor=extractor, parameters={}, config={})

    response = requests.Response()

    response.headers = {"A_HEADER": "HEADER_VALUE"}
    response_body = {"next": "https://airbyte.io/next_url", "page_metadata": {"limit": 5}}
    response._content = json.dumps(response_body).encode("utf-8")

    next_page_token = paginator_strategy.next_page_token(response, 2, None, 4)
    assert next_page_token is None

    # Validate that the PaginationStrategy is stateless and calling next_page_token() again returns the same result
    next_page_token = paginator_strategy.next_page_token(response, 2, None, 4)
    assert next_page_token is None


def test_offset_increment_paginator_strategy_rises():
    paginator_strategy = OffsetIncrement(
        page_size="{{ parameters['page_size'] }}",
        extractor=DpathExtractor(field_path=["results"], parameters={}, config={}),
        parameters={"page_size": "invalid value"},
        config={},
    )
    with pytest.raises(Exception) as exc:
        paginator_strategy.get_page_size()
    assert str(exc.value) == "invalid value is of type <class 'str'>. Expected <class 'int'>"


@pytest.mark.parametrize(
    "inject_on_first_request, expected_initial_token",
    [
        pytest.param(True, 0, id="test_with_inject_offset"),
        pytest.param(False, None, id="test_without_inject_offset"),
    ],
)
def test_offset_increment_paginator_strategy_initial_token(
    inject_on_first_request: bool, expected_initial_token: Optional[Any]
):
    extractor = DpathExtractor(field_path=[""], parameters={}, config={})
    paginator_strategy = OffsetIncrement(
        page_size=20,
        extractor=extractor,
        parameters={},
        config={},
        inject_on_first_request=inject_on_first_request,
    )

    assert paginator_strategy.initial_token == expected_initial_token
