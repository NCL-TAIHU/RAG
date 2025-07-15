import unittest
from src.core.router import Router, SimpleRouter, SparsityRouter
from src.core.search_engine import SearchSpec
from unittest.mock import MagicMock

# Minimal mock Filter class for testing
class MockFilter:
    def __init__(self, must_fields=None, **kwargs):
        self._must_fields = must_fields or []
        for k, v in kwargs.items():
            setattr(self, k, v)
    def must_fields(self):
        return self._must_fields

class TestRouter(unittest.TestCase):
    def setUp(self):
        # Create two SearchSpecs: one optimal for 'strong', one for 'weak'
        self.strong_spec = SearchSpec(name="strong_engine", optimal_for="strong")
        self.weak_spec = SearchSpec(name="weak_engine", optimal_for="weak")
        self.specs = [self.strong_spec, self.weak_spec]

    def test_simple_router_always_returns_zero(self):
        router = SimpleRouter(self.specs)
        filter = MockFilter()
        self.assertEqual(router.route(filter), 0)

    def test_sparsity_router_prefers_strong_when_field_present(self):
        router = SparsityRouter(self.specs)
        # Filter with a must_field set to a non-None value
        filter = MockFilter(must_fields=["foo"], foo="bar")
        self.assertEqual(router.route(filter), 0)  # strong_spec index

    def test_sparsity_router_fallback_to_weak(self):
        router = SparsityRouter(self.specs)
        # Filter with must_field present but no strong engine
        specs = [self.weak_spec]
        router = SparsityRouter(specs)
        filter = MockFilter(must_fields=["foo"], foo="bar")
        self.assertEqual(router.route(filter), 0)  # only weak_spec

    def test_sparsity_router_returns_dense_when_no_must_fields(self):
        router = SparsityRouter(self.specs)
        filter = MockFilter(must_fields=["foo"], foo=None)
        self.assertEqual(router.route(filter), 1)  # weak_spec index

    def test_router_from_default(self):
        router = Router.from_default("simple", self.specs)
        self.assertIsInstance(router, SimpleRouter)
        router = Router.from_default("sparsity", self.specs)
        self.assertIsInstance(router, SparsityRouter)
        with self.assertRaises(ValueError):
            Router.from_default("unknown", self.specs)

    def test_router_requires_at_least_one_spec(self):
        with self.assertRaises(AssertionError):
            SimpleRouter([])

if __name__ == "__main__":
    unittest.main() 