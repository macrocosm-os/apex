# ruff: noqa: E402
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared import settings

settings.shared_settings = settings.SharedSettings(mode="mock")
from prompting.rewards.web_retrieval import WebRetrievalRewardModel


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "creation_date, expected_age, domain",
    [
        # Domain created 100 days ago.
        (datetime.now() - timedelta(days=100), 100, "testdomain100.com"),
        # Domain created 10 days ago.
        (datetime.now() - timedelta(days=10), 10, "testdomain10.com"),
        # Domain has no valid creation_date => fallback_age.
        (None, 1_000_000, "testdomainnone.com"),
    ],
)
async def test_domain_age(creation_date: datetime, expected_age: int, domain: str):
    mock_whois = MagicMock()
    mock_whois.creation_date = creation_date
    with patch("prompting.rewards.web_retrieval.whois.whois", return_value=mock_whois):
        age = await WebRetrievalRewardModel.domain_age_days(domain, fallback_age=1_000_000)
        assert age == expected_age


@pytest.mark.parametrize(
    "completion, expected_url, expected_content, expected_relevant",
    [
        (
            '{"url": "http://example.com", "content": "This is some content.", "relevant": "Section 1"}',
            "http://example.com",
            "This is some content.",
            "Section 1",
        ),
        # Invalid JSON should return an empty list
        ("Invalid JSON string", None, None, None),
    ],
)
def test_parse_response(completion, expected_url, expected_content, expected_relevant):
    response = WebRetrievalRewardModel._parse_response(completion)

    if not response:  # empty list => invalid JSON
        assert expected_url is None
        assert expected_content is None
        assert expected_relevant is None
    else:
        # For the valid test case, we expect exactly one WebsiteResult
        assert len(response) == 1
        assert response[0].url == expected_url
        assert response[0].content == expected_content
        assert response[0].relevant == expected_relevant


async def test_cosine_similarity_identical_embeddings():
    # Mock identical embeddings.
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = np.array([1, 2, 3])

    model = WebRetrievalRewardModel()
    model.embedding_model = mock_embedding_model

    similarity = await model._cosine_similarity("content1", "content1")
    assert similarity == pytest.approx(1.0)


async def test_cosine_similarity_orthogonal_embeddings():
    # Mock orthogonal embeddings.
    def encode_mock(text, to_numpy):
        return np.array([1, 0]) if text == "content1" else np.array([0, 1])

    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.side_effect = encode_mock

    model = WebRetrievalRewardModel()
    model.embedding_model = mock_embedding_model

    similarity = await model._cosine_similarity("content1", "content2")
    assert similarity == pytest.approx(0.0)


# TODO: Implement reward tests.
# @patch("trafilatura.fetch_url")
# @patch("trafilatura.extract")
# def test_reward(mock_extract, mock_fetch_url):
#     # Mock the embedding_model.encode method.
#     def encode_mock(text, to_numpy):
#         if text == "test search":
#             return np.array([1, 0, 0])
#         elif text == "Response content":
#             return np.array([0.9, 0.1, 0])
#         elif text == "Extracted content from URL":
#             return np.array([0.8, 0.2, 0])
#         elif text == "Reference website content":
#             # Adjusted embedding to ensure |search_response_sim - search_reference_sim| <= 0.4
#             # For example, cosine similarity with [1,0,0] is 0.3, so search_reference_sim = 0.7
#             return np.array([0.3, 0.95394, 0])
#         elif text == "Section 1":
#             # Providing a non-zero embedding to avoid NaN in cosine similarity
#             return np.array([0, 1, 0])
#         else:
#             return np.array([0, 0, 0])

#     # Mock the embedding_model.
#     embedding_model_mock = MagicMock()
#     embedding_model_mock.encode.side_effect = encode_mock

#     # Assign the mock embedding model to the model.
#     model = WebRetrievalRewardModel()
#     model.embedding_model = embedding_model_mock

#     # Mock trafilatura functions.
#     mock_fetch_url.return_value = "<html><body>Webpage content</body></html>"
#     mock_extract.return_value = "Extracted content from URL"

#     # Create a reference.
#     reference_entry = DDGDatasetEntry(
#         search_term="test search",
#         website_url="http://example.com",
#         website_content="Reference website content"
#     )
#     reference_json = reference_entry.model_dump_json()

#     # Create a response_event with JSON completion.
#     completion = json.dumps({
#         "url": "http://example.com",
#         "content": "Response content",
#         "relevant": "Section 1"
#     })
#     synapse = StreamPromptingSynapse(
#             task_name="web_retrieval",
#             roles=["assistant"],
#             messages=["Hello"],
#             completion=completion,
#         )
#     result = SynapseStreamResult(synapse=synapse)
#     response_event = DendriteResponseEvent(
#         uids=[0.0],
#         timeout=0,
#         stream_results=[result],
#     )

#     # Call the reward function.
#     output = model.reward(reference=reference_json, response_event=response_event)

#     # Expected score calculation.
#     search_response_sim = 1.0 - spatial.distance.cosine(
#         np.array([1, 0, 0]),
#         np.array([0.9, 0.1, 0])
#     )  # ≈ 0.993
#     valid_url_score = 1.0 - spatial.distance.cosine(
#         np.array([0.9, 0.1, 0]),
#         np.array([0.8, 0.2, 0])
#     )  # ≈ 0.991
#     search_relevant_sim = 1.0 - spatial.distance.cosine(
#         np.array([1, 0, 0]),
#         np.array([0, 1, 0])
#     )  # = 1.0
#     expected_score = (search_response_sim + valid_url_score + search_relevant_sim) / 3  # ≈ 0.992

#     assert isinstance(output, BatchRewardOutput)
#     assert isinstance(output.rewards, np.ndarray)
#     assert output.rewards.shape == (1,)
#     assert output.rewards[0] == pytest.approx(expected_score, abs=1e-3)


# @patch("trafilatura.fetch_url")
# @patch("trafilatura.extract")
# def test_reward_response_reference_ratio_exceeds_threshold(mock_extract, mock_fetch_url):
#     # Mock the embedding_model.encode method.
#     def encode_mock(text, to_numpy):
#         if text == "test search":
#             return np.array([1, 0, 0])
#         elif text == "Response content":
#             return np.array([0.5, 0.5, 0])
#         elif text == "Reference website content":
#             return np.array([0.95, 0.05, 0])
#         else:
#             return np.array([0, 0, 0])

#     embedding_model_mock = MagicMock()
#     embedding_model_mock.encode.side_effect = encode_mock

#     model = WebRetrievalRewardModel()
#     model.embedding_model = embedding_model_mock

#     mock_fetch_url.return_value = "<html><body>Webpage content</body></html>"
#     mock_extract.return_value = "Extracted content from URL"

#     reference_entry = DDGDatasetEntry(
#         search_term="test search",
#         website_url="http://example.com",
#         website_content="Reference website content"
#     )
#     reference_json = reference_entry.model_dump_json()

#     completion = json.dumps({
#         "url": "http://example.com",
#         "content": "Response content",
#         "relevant": "Section 1"
#     })
#     synapse = StreamPromptingSynapse(
#             task_name="web_retrieval",
#             roles=["assistant"],
#             messages=["Hello"],
#             completion=completion,
#         )
#     result = SynapseStreamResult(synapse=synapse)
#     response_event = DendriteResponseEvent(
#         uids=[0.0],
#         timeout=0,
#         stream_results=[result],
#     )

#     output = model.reward(reference=reference_json, response_event=response_event)
#     assert output.rewards[0] == 0


# @patch("trafilatura.fetch_url")
# @patch("trafilatura.extract")
# def test_reward_valid_url_score_below_threshold(mock_extract, mock_fetch_url):
#     # Mock the embedding_model.encode method.
#     def encode_mock(text, to_numpy):
#         if text == "Response content":
#             return np.array([1, 0, 0])
#         elif text == "Extracted content from URL":
#             return np.array([0, 1, 0])
#         elif text == "test search":
#             return np.array([1, 0, 0])
#         elif text == "Reference website content":
#             return np.array([1, 0, 0])
#         else:
#             return np.array([0, 0, 0])

#     embedding_model_mock = MagicMock()
#     embedding_model_mock.encode.side_effect = encode_mock

#     model = WebRetrievalRewardModel()
#     model.embedding_model = embedding_model_mock

#     mock_fetch_url.return_value = "<html><body>Webpage content</body></html>"
#     mock_extract.return_value = "Extracted content from URL"

#     reference_entry = DDGDatasetEntry(
#         search_term="test search",
#         website_url="http://example.com",
#         website_content="Reference website content"
#     )
#     reference_json = reference_entry.model_dump_json()

#     completion = json.dumps({
#         "url": "http://example.com",
#         "content": "Response content",
#         "relevant": "Section 1"
#     })
#     synapse = StreamPromptingSynapse(
#             task_name="web_retrieval",
#             roles=["assistant"],
#             messages=["Hello"],
#             completion=completion,
#         )
#     result = SynapseStreamResult(synapse=synapse)
#     response_event = DendriteResponseEvent(
#         uids=[0.0],
#         timeout=0,
#         stream_results=[result],
#     )

#     output = model.reward(reference=reference_json, response_event=response_event)
#     assert output.rewards[0] == 0
