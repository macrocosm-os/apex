import httpx
import json

from cli.utils.wallet import load_keypair_from_file
from common.utils.epistula import generate_header, create_message_body
from common.models.api.code import CodeRequest, CodeResponse
from common.models.api.submission import SubmissionDetail, FileRequest
from common.models.api.file import ChunkedFileData
from common.models.api.competition import CompetitionRequest, CompetitionResponse
from common.settings import ORCHESTRATOR_SCHEMA, ORCHESTRATOR_HOST, ORCHESTRATOR_PORT


class Client:
    def __init__(self, hotkey_file_path: str, timeout: float = 60.0):
        print(f"ORCHESTRATOR_SCHEMA: {ORCHESTRATOR_SCHEMA}")
        print(f"ORCHESTRATOR_HOST: {ORCHESTRATOR_HOST}")
        print(f"ORCHESTRATOR_PORT: {ORCHESTRATOR_PORT}")
        if ORCHESTRATOR_SCHEMA == "https":
            self.url = f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}"
        else:
            self.url = f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"
        self.client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        self.keypair = load_keypair_from_file(hotkey_file_path)

    async def _make_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        params: dict | None = None,
        additional_headers: dict | None = None,
    ) -> httpx.Response:
        try:
            if self.keypair:
                # For GET requests with params, sign the full params data (before filtering)
                # For POST requests with body, sign the body data
                sign_data = params if params is not None else (body if body is not None else {})
                body_bytes = create_message_body(data=sign_data)
                headers = generate_header(self.keypair, body_bytes)
            else:
                headers = {}
        except Exception as e:
            raise

        if additional_headers:
            headers.update(additional_headers)

        try:
            # Use params for query parameters (GET requests) and json for body (POST requests)
            request_kwargs = {"headers": headers}
            if params is not None:
                # Filter out None values for query parameters, but we signed the full params above
                filtered_params = {k: v for k, v in params.items() if v is not None}
                request_kwargs["params"] = filtered_params
            if body is not None:
                request_kwargs["json"] = body

            response = await self.client.request(method, f"{self.url}{path}", **request_kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            # Enhanced error reporting for HTTP status errors
            error_msg = f"HTTP Error {e.response.status_code}"
            try:
                error_detail = e.response.text
                try:
                    error_detail = json.loads(error_detail).get("detail")
                except Exception:
                    pass
                if error_detail:
                    error_msg += f": {error_detail}"
            except Exception:
                pass
            raise httpx.HTTPStatusError(error_msg, request=e.request, response=e.response)
        except httpx.RequestError as e:
            # Enhanced error reporting for request errors
            print(f"failed to connect to {self.url}")
            raise httpx.RequestError(f"Request failed: {e}")

    async def submit_solution(self, solution: CodeRequest) -> CodeResponse | None:
        """Submit a solution to the orchestrator"""
        try:
            response = await self._make_request(
                method="POST",
                path="/miner/submission",
                body=solution.model_dump(),
            )
            return CodeResponse.model_validate(response.json())

        except Exception as e:
            raise

    async def get_submission_detail(self, submission_id: int) -> SubmissionDetail | None:
        """Get detailed submission information including eval metadata and files."""
        try:
            # For GET requests, we need to use query parameters instead of body
            params = {"submission_id": submission_id}

            # Generate authentication headers
            if self.keypair:
                body_bytes = create_message_body(data=params)
                headers = generate_header(self.keypair, body_bytes)
            else:
                headers = {}

            response = await self.client.get(
                f"{self.url}/miner/submission/{submission_id}/detail", params=params, headers=headers
            )
            response.raise_for_status()
            return SubmissionDetail.model_validate(response.json())
        except Exception as e:
            raise

    async def get_submission_code(self, code_request: CodeRequest) -> CodeResponse | None:
        """Get submission code using the code endpoint."""
        try:
            params = code_request.model_dump()
            response = await self._make_request(
                method="GET",
                path="/miner/submission/code",
                params=params,
            )
            return CodeResponse.model_validate(response.json())
        except Exception as e:
            raise

    async def get_file_chunked(self, file_request: FileRequest) -> ChunkedFileData | None:
        """Get chunked file data for a submission file."""
        try:
            params = file_request.model_dump()
            response = await self._make_request(
                method="GET",
                path="/miner/submission/file",
                params=params,
            )
            return ChunkedFileData.model_validate(response.json())
        except Exception as e:
            raise

    async def list_competitions(
        self,
        id: int | None = None,
        name: str | None = None,
        pkg: str | None = None,
        ptype: str | None = None,
        ctype: str | None = None,
        state: str | None = None,
        start_idx: int = 0,
        count: int = 10,
    ) -> CompetitionResponse:
        """List competitions using signed GET request."""
        req = CompetitionRequest(
            id=id,
            name=name,
            pkg=pkg,
            ptype=ptype,
            ctype=ctype,
            state=state,
            start_idx=start_idx,
            count=count,
        )

        sign_payload = req.model_dump()
        params = {k: v for k, v in sign_payload.items() if v is not None}

        headers = generate_header(self.keypair, create_message_body(sign_payload)) if self.keypair else {}

        response = await self.client.get(f"{self.url}/miner/competition", params=params, headers=headers)
        response.raise_for_status()
        return CompetitionResponse.model_validate(response.json())

    async def close(self):
        """Close the httpx client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures client is closed."""
        await self.close()
