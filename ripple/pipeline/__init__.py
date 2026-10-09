"""Public facade for the typed RIPPLe route dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ripple.scientist.router import (
    load_pipeline_request,
    plan_pipeline_route,
    run_pipeline_route,
)
from ripple.scientist.schemas.orchestration import PipelineRunRequest
from ripple.scientist.schemas.routes import PipelineRouteResult, RoutePlan


class PipelineOrchestrator:
    """Plan and execute one validated pipeline request at a time."""

    def __init__(
        self,
        *,
        researcher_agent_provider: Literal["bedrock", "openai"] | None = None,
    ) -> None:
        if researcher_agent_provider not in {None, "bedrock", "openai"}:
            raise ValueError(
                "researcher_agent_provider must be bedrock, openai, or None"
            )
        self.researcher_agent_provider = researcher_agent_provider

    @staticmethod
    def load_request(path: Path) -> PipelineRunRequest:
        return load_pipeline_request(path)

    @staticmethod
    def plan(
        request: PipelineRunRequest,
        *,
        simulation_configuration: Path | None = None,
        preprocessing_output_root: Path | None = None,
        researcher_output_root: Path | None = None,
        repository_root: Path | None = None,
    ) -> RoutePlan:
        return plan_pipeline_route(
            request,
            simulation_configuration=simulation_configuration,
            preprocessing_output_root=preprocessing_output_root,
            researcher_output_root=researcher_output_root,
            repository_root=repository_root,
        )

    async def run(
        self,
        request: PipelineRunRequest,
        *,
        request_base_directory: Path,
        simulation_configuration: Path | None = None,
        preprocessing_output_root: Path | None = None,
        researcher_output_root: Path | None = None,
        repository_root: Path | None = None,
        campaign_run_id: str | None = None,
    ) -> PipelineRouteResult:
        return await run_pipeline_route(
            request,
            request_base_directory=request_base_directory,
            simulation_configuration=simulation_configuration,
            preprocessing_output_root=preprocessing_output_root,
            researcher_output_root=researcher_output_root,
            repository_root=repository_root,
            campaign_run_id=campaign_run_id,
            researcher_agent_provider=self.researcher_agent_provider,
        )


__all__ = ["PipelineOrchestrator"]
