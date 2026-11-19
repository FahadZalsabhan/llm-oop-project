"""Pricing plans that can be shared between models (aggregation)."""


class PricingPlan:
    """A pricing plan that defines the cost per 1000 tokens."""

    def __init__(self, plan_name: str, price_per_1000_tokens: float) -> None:
        """Create a pricing plan with a name and a price per 1000 tokens."""
        if price_per_1000_tokens < 0:
            raise ValueError("Price per 1000 tokens cannot be negative.")
        self.plan_name = plan_name
        self.price_per_1000_tokens = price_per_1000_tokens

    def calculate_cost(self, tokens: int) -> float:
        """Return the cost of the given number of tokens."""
        if tokens < 0:
            raise ValueError("Number of tokens cannot be negative.")
        return tokens / 1000 * self.price_per_1000_tokens

    def __str__(self) -> str:
        """Return a readable description of the plan."""
        return f"{self.plan_name} plan (${self.price_per_1000_tokens} per 1000 tokens)"

    def __repr__(self) -> str:
        """Return a technical representation of the plan."""
        return (
            f"PricingPlan(plan_name='{self.plan_name}', "
            f"price_per_1000_tokens={self.price_per_1000_tokens})"
        )
