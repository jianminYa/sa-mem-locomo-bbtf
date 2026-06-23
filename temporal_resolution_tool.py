"""
Temporal Resolution Tool for LLM Function Calling

This module provides a function that LLMs can call to resolve relative temporal expressions
into absolute dates, ensuring accurate date calculations including leap years.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List


def resolve_temporal_expression(
    observation_time: str,
    expression: str
) -> Dict[str, Any]:
    """
    Resolve a relative temporal expression to absolute date(s).

    Args:
        observation_time: The reference time (session_end) in format "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
        expression: The relative temporal expression (e.g., "yesterday", "last week", "last Thursday")

    Returns:
        Dictionary with:
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format (same as start_date for point-in-time)
        - day_of_week: Day of week for the observation_time
        - resolved_expression: Human-readable description

    Examples:
        >>> resolve_temporal_expression("2026-03-04", "yesterday")
        {'start_date': '2026-03-03', 'end_date': '2026-03-03', 'day_of_week': 'Wednesday', 'resolved_expression': 'yesterday (2026-03-03)'}

        >>> resolve_temporal_expression("2026-03-04", "last Thursday")
        {'start_date': '2026-02-26', 'end_date': '2026-02-26', 'day_of_week': 'Wednesday', 'resolved_expression': 'last Thursday (2026-02-26)'}

        >>> resolve_temporal_expression("2026-03-04", "last week")
        {'start_date': '2026-02-26', 'end_date': '2026-03-04', 'day_of_week': 'Wednesday', 'resolved_expression': 'last week (2026-02-26 to 2026-03-04)'}
    """
    # Parse observation time
    if 'T' in observation_time:
        obs_dt = datetime.fromisoformat(observation_time.replace('Z', '+00:00'))
    else:
        obs_dt = datetime.strptime(observation_time, "%Y-%m-%d")

    obs_day_of_week = obs_dt.strftime("%A")
    expr_lower = expression.lower().strip()

    # Point-in-time expressions
    if expr_lower == "yesterday":
        target_dt = obs_dt - timedelta(days=1)
        return {
            "start_date": target_dt.strftime("%Y-%m-%d"),
            "end_date": target_dt.strftime("%Y-%m-%d"),
            "day_of_week": obs_day_of_week,
            "resolved_expression": f"yesterday ({target_dt.strftime('%Y-%m-%d')})"
        }

    if expr_lower == "today":
        return {
            "start_date": obs_dt.strftime("%Y-%m-%d"),
            "end_date": obs_dt.strftime("%Y-%m-%d"),
            "day_of_week": obs_day_of_week,
            "resolved_expression": f"today ({obs_dt.strftime('%Y-%m-%d')})"
        }

    if expr_lower in ["this morning", "this afternoon", "this evening", "tonight"]:
        return {
            "start_date": obs_dt.strftime("%Y-%m-%d"),
            "end_date": obs_dt.strftime("%Y-%m-%d"),
            "day_of_week": obs_day_of_week,
            "resolved_expression": f"{expression} ({obs_dt.strftime('%Y-%m-%d')})"
        }

    # Last [weekday]
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    for day_name, day_num in weekdays.items():
        if f"last {day_name}" in expr_lower:
            current_weekday = obs_dt.weekday()
            # Calculate days back to last occurrence of target weekday
            if current_weekday >= day_num:
                days_back = current_weekday - day_num
                if days_back == 0:
                    days_back = 7  # If today is the target day, go back a full week
            else:
                days_back = 7 - (day_num - current_weekday)

            target_dt = obs_dt - timedelta(days=days_back)
            return {
                "start_date": target_dt.strftime("%Y-%m-%d"),
                "end_date": target_dt.strftime("%Y-%m-%d"),
                "day_of_week": obs_day_of_week,
                "resolved_expression": f"last {day_name.capitalize()} ({target_dt.strftime('%Y-%m-%d')})"
            }

    # Duration expressions
    if expr_lower == "last week":
        # Last week = 7 days ending at observation time
        start_dt = obs_dt - timedelta(days=7)
        return {
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": obs_dt.strftime("%Y-%m-%d"),
            "day_of_week": obs_day_of_week,
            "resolved_expression": f"last week ({start_dt.strftime('%Y-%m-%d')} to {obs_dt.strftime('%Y-%m-%d')})"
        }

    if expr_lower == "last month":
        # Last month = previous calendar month
        if obs_dt.month == 1:
            last_month_year = obs_dt.year - 1
            last_month_num = 12
        else:
            last_month_year = obs_dt.year
            last_month_num = obs_dt.month - 1

        start_dt = datetime(last_month_year, last_month_num, 1)

        # Calculate last day of last month
        if last_month_num == 12:
            next_month = datetime(last_month_year + 1, 1, 1)
        else:
            next_month = datetime(last_month_year, last_month_num + 1, 1)
        end_dt = next_month - timedelta(days=1)

        return {
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "day_of_week": obs_day_of_week,
            "resolved_expression": f"last month ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})"
        }

    if expr_lower == "last year":
        last_year = obs_dt.year - 1
        start_dt = datetime(last_year, 1, 1)
        end_dt = datetime(last_year, 12, 31)
        return {
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "day_of_week": obs_day_of_week,
            "resolved_expression": f"last year ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})"
        }

    # If no match, return observation time as fallback
    return {
        "start_date": obs_dt.strftime("%Y-%m-%d"),
        "end_date": obs_dt.strftime("%Y-%m-%d"),
        "day_of_week": obs_day_of_week,
        "resolved_expression": f"{expression} ({obs_dt.strftime('%Y-%m-%d')})",
        "warning": f"Could not parse expression '{expression}', using observation time as fallback"
    }


# OpenAI Function Calling Schema
TEMPORAL_RESOLUTION_FUNCTION_SCHEMA = {
    "name": "resolve_temporal_expression",
    "description": "Resolve a relative temporal expression (like 'yesterday', 'last week', 'last Thursday') to absolute date(s). Use this function whenever you encounter relative time expressions in the dialog to ensure accurate date calculations.",
    "parameters": {
        "type": "object",
        "properties": {
            "observation_time": {
                "type": "string",
                "description": "The reference time (session_end) in format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'. This is the 'current time' from which to calculate relative dates."
            },
            "expression": {
                "type": "string",
                "description": "The relative temporal expression to resolve, e.g., 'yesterday', 'last week', 'last Thursday', 'this morning', 'last month', 'last year'"
            }
        },
        "required": ["observation_time", "expression"]
    }
}


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("2026-03-04", "yesterday"),
        ("2026-03-04", "last Thursday"),
        ("2026-03-04", "last week"),
        ("2026-03-04", "this morning"),
        ("2026-03-04", "last month"),
        ("2024-03-01", "last month"),  # Test February in leap year
        ("2025-03-01", "last month"),  # Test February in non-leap year
    ]

    print("Temporal Resolution Function Test Cases:\n")
    for obs_time, expr in test_cases:
        result = resolve_temporal_expression(obs_time, expr)
        print(f"Observation: {obs_time} ({result['day_of_week']})")
        print(f"Expression: '{expr}'")
        print(f"Result: {result['resolved_expression']}")
        if 'warning' in result:
            print(f"Warning: {result['warning']}")
        print()
