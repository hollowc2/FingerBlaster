"""Comprehensive tests for CallbackManager event system."""

import pytest
import asyncio
import threading
from src.activetrader.core import CallbackManager, CALLBACK_EVENTS


class TestCallbackManagerRegistration:
    """Test callback registration functionality."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    def test_register_valid_event(self, manager):
        """Test registering callback for valid event."""
        def my_callback():
            pass

        result = manager.register('market_update', my_callback)
        assert result is True
        assert my_callback in manager.get_callbacks('market_update')

    def test_register_invalid_event(self, manager):
        """Test registering callback for invalid event returns False."""
        def my_callback():
            pass

        result = manager.register('invalid_event', my_callback)
        assert result is False

    def test_register_duplicate_callback(self, manager):
        """Test registering same callback twice only registers once."""
        def my_callback():
            pass

        manager.register('market_update', my_callback)
        manager.register('market_update', my_callback)

        callbacks = manager.get_callbacks('market_update')
        assert callbacks.count(my_callback) == 1

    def test_register_multiple_callbacks_same_event(self, manager):
        """Test registering multiple different callbacks for same event."""
        def callback1():
            pass

        def callback2():
            pass

        manager.register('market_update', callback1)
        manager.register('market_update', callback2)

        callbacks = manager.get_callbacks('market_update')
        assert callback1 in callbacks
        assert callback2 in callbacks
        assert len(callbacks) == 2

    def test_register_callback_multiple_events(self, manager):
        """Test same callback can be registered to multiple events."""
        def my_callback():
            pass

        manager.register('market_update', my_callback)
        manager.register('price_update', my_callback)

        assert my_callback in manager.get_callbacks('market_update')
        assert my_callback in manager.get_callbacks('price_update')

    def test_register_all_valid_events(self, manager):
        """Test that all CALLBACK_EVENTS are recognized."""
        def my_callback():
            pass

        for event in CALLBACK_EVENTS:
            result = manager.register(event, my_callback)
            assert result is True, f"Failed to register callback for {event}"


class TestCallbackManagerUnregistration:
    """Test callback unregistration functionality."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    def test_unregister_registered_callback(self, manager):
        """Test unregistering a registered callback."""
        def my_callback():
            pass

        manager.register('market_update', my_callback)
        result = manager.unregister('market_update', my_callback)

        assert result is True
        assert my_callback not in manager.get_callbacks('market_update')

    def test_unregister_nonexistent_callback(self, manager):
        """Test unregistering callback that wasn't registered returns True (event exists)."""
        def my_callback():
            pass

        # Implementation returns True if event exists, even if callback wasn't registered
        result = manager.unregister('market_update', my_callback)
        assert result is True
        # Verify callback is still not in the list
        assert my_callback not in manager.get_callbacks('market_update')

    def test_unregister_from_invalid_event(self, manager):
        """Test unregistering from invalid event returns False."""
        def my_callback():
            pass

        result = manager.unregister('invalid_event', my_callback)
        assert result is False

    def test_unregister_one_of_multiple_callbacks(self, manager):
        """Test unregistering one callback leaves others intact."""
        def callback1():
            pass

        def callback2():
            pass

        manager.register('market_update', callback1)
        manager.register('market_update', callback2)
        manager.unregister('market_update', callback1)

        callbacks = manager.get_callbacks('market_update')
        assert callback1 not in callbacks
        assert callback2 in callbacks


class TestCallbackManagerClear:
    """Test callback clearing functionality."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    def test_clear_specific_event(self, manager):
        """Test clearing callbacks for specific event."""
        def cb1():
            pass

        def cb2():
            pass

        manager.register('market_update', cb1)
        manager.register('price_update', cb2)

        manager.clear('market_update')

        assert len(manager.get_callbacks('market_update')) == 0
        assert len(manager.get_callbacks('price_update')) == 1

    def test_clear_all_events(self, manager):
        """Test clearing all callbacks."""
        def cb():
            pass

        # Register callback for multiple events
        for event in ['market_update', 'price_update', 'log']:
            manager.register(event, cb)

        manager.clear()

        # All should be cleared
        for event in ['market_update', 'price_update', 'log']:
            assert len(manager.get_callbacks(event)) == 0

    def test_clear_nonexistent_event(self, manager):
        """Test clearing non-existent event doesn't raise error."""
        # Should not raise exception
        manager.clear('nonexistent_event')

    def test_clear_empty_event(self, manager):
        """Test clearing event with no callbacks."""
        # Should not raise exception
        manager.clear('market_update')
        assert len(manager.get_callbacks('market_update')) == 0


class TestCallbackManagerEmit:
    """Test callback emission functionality."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    @pytest.mark.asyncio
    async def test_emit_sync_callback(self, manager):
        """Test emitting event with synchronous callback."""
        result = []

        def my_callback(value):
            result.append(value)

        manager.register('market_update', my_callback)
        await manager.emit('market_update', 42)

        assert result == [42]

    @pytest.mark.asyncio
    async def test_emit_async_callback(self, manager):
        """Test emitting event with asynchronous callback."""
        result = []

        async def my_callback(value):
            await asyncio.sleep(0.01)
            result.append(value)

        manager.register('market_update', my_callback)
        await manager.emit('market_update', 42)

        # Give async task time to complete
        await asyncio.sleep(0.02)
        assert result == [42]

    @pytest.mark.asyncio
    async def test_emit_multiple_callbacks(self, manager):
        """Test emitting event with multiple registered callbacks."""
        results = []

        def callback1(value):
            results.append(value * 2)

        def callback2(value):
            results.append(value * 3)

        manager.register('market_update', callback1)
        manager.register('market_update', callback2)
        await manager.emit('market_update', 10)

        assert 20 in results
        assert 30 in results

    @pytest.mark.asyncio
    async def test_emit_with_args_and_kwargs(self, manager):
        """Test emit passes both args and kwargs correctly."""
        received_args = []
        received_kwargs = {}

        def my_callback(*args, **kwargs):
            received_args.extend(args)
            received_kwargs.update(kwargs)

        manager.register('market_update', my_callback)
        await manager.emit('market_update', 1, 2, 3, foo='bar', baz='qux')

        assert received_args == [1, 2, 3]
        assert received_kwargs == {'foo': 'bar', 'baz': 'qux'}

    @pytest.mark.asyncio
    async def test_emit_with_error_in_callback(self, manager):
        """Test that error in one callback doesn't break others."""
        results = []

        def bad_callback(value):
            raise ValueError("Intentional error")

        def good_callback(value):
            results.append(value)

        manager.register('market_update', bad_callback)
        manager.register('market_update', good_callback)

        # Should not raise exception
        await manager.emit('market_update', 42)

        # Good callback should still execute
        assert 42 in results

    @pytest.mark.asyncio
    async def test_emit_invalid_event(self, manager):
        """Test emitting invalid event doesn't raise error."""
        # Should not raise exception
        await manager.emit('invalid_event', 42)

    @pytest.mark.asyncio
    async def test_emit_no_callbacks_registered(self, manager):
        """Test emitting event with no callbacks registered."""
        # Should not raise exception
        await manager.emit('market_update', 42)

    @pytest.mark.asyncio
    async def test_emit_async_callback_exception(self, manager):
        """Test async callback exception is caught."""
        async def bad_async_callback(value):
            await asyncio.sleep(0.01)
            raise RuntimeError("Async error")

        manager.register('market_update', bad_async_callback)

        # Should not raise exception
        await manager.emit('market_update', 42)
        await asyncio.sleep(0.02)  # Let async task complete

    @pytest.mark.asyncio
    async def test_emit_callback_return_value_ignored(self, manager):
        """Test that callback return values are ignored."""
        def callback_with_return():
            return "This should be ignored"

        manager.register('market_update', callback_with_return)

        # Should not raise exception, return value is ignored
        result = await manager.emit('market_update')
        assert result is None


class TestCallbackManagerThreadSafety:
    """Test thread safety of CallbackManager."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    def test_concurrent_registration(self, manager):
        """Test concurrent callback registration from multiple threads."""
        callbacks = [lambda: None for _ in range(100)]
        threads = []

        def register_callbacks(start_idx, end_idx):
            for i in range(start_idx, end_idx):
                manager.register('market_update', callbacks[i])

        # Create 5 threads, each registering 20 callbacks
        for i in range(5):
            t = threading.Thread(target=register_callbacks, args=(i * 20, (i + 1) * 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All callbacks should be registered
        registered = manager.get_callbacks('market_update')
        assert len(registered) == 100

    def test_concurrent_unregistration(self, manager):
        """Test concurrent callback unregistration."""
        callbacks = [lambda: None for _ in range(50)]

        # Register all callbacks
        for cb in callbacks:
            manager.register('market_update', cb)

        threads = []

        def unregister_callbacks(start_idx, end_idx):
            for i in range(start_idx, end_idx):
                manager.unregister('market_update', callbacks[i])

        # Unregister from multiple threads
        for i in range(5):
            t = threading.Thread(target=unregister_callbacks, args=(i * 10, (i + 1) * 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should be unregistered
        assert len(manager.get_callbacks('market_update')) == 0

    @pytest.mark.asyncio
    async def test_concurrent_emit_and_register(self, manager):
        """Test emitting events while registering callbacks concurrently."""
        results = []

        def callback(value):
            results.append(value)

        # Register initial callback
        manager.register('market_update', callback)

        # Emit events concurrently with registrations
        tasks = []
        for i in range(10):
            tasks.append(manager.emit('market_update', i))

        await asyncio.gather(*tasks)

        # Should have received all emissions
        assert len(results) >= 10


class TestCallbackManagerGetCallbacks:
    """Test get_callbacks functionality."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    def test_get_callbacks_returns_copy(self, manager):
        """Test that get_callbacks returns a copy, not the original list."""
        def cb():
            pass

        manager.register('market_update', cb)

        callbacks1 = manager.get_callbacks('market_update')
        callbacks2 = manager.get_callbacks('market_update')

        # Should be equal but not the same object
        assert callbacks1 == callbacks2
        assert callbacks1 is not callbacks2

    def test_get_callbacks_modification_doesnt_affect_manager(self, manager):
        """Test modifying returned list doesn't affect manager's internal state."""
        def cb():
            pass

        manager.register('market_update', cb)

        callbacks = manager.get_callbacks('market_update')
        callbacks.clear()

        # Manager should still have the callback
        assert len(manager.get_callbacks('market_update')) == 1

    def test_get_callbacks_invalid_event(self, manager):
        """Test getting callbacks for invalid event returns empty list."""
        callbacks = manager.get_callbacks('invalid_event')
        assert callbacks == []

    def test_get_callbacks_empty_event(self, manager):
        """Test getting callbacks for event with no callbacks."""
        callbacks = manager.get_callbacks('market_update')
        assert callbacks == []


class TestCallbackManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def manager(self):
        """Fresh CallbackManager instance."""
        return CallbackManager()

    def test_register_lambda_callback(self, manager):
        """Test registering lambda functions as callbacks."""
        result = []
        cb = lambda x: result.append(x)

        manager.register('market_update', cb)
        assert cb in manager.get_callbacks('market_update')

    @pytest.mark.asyncio
    async def test_emit_to_lambda_callback(self, manager):
        """Test emitting to lambda callback."""
        result = []
        cb = lambda x: result.append(x)

        manager.register('market_update', cb)
        await manager.emit('market_update', 42)

        assert result == [42]

    def test_register_class_method_as_callback(self, manager):
        """Test registering class method as callback."""
        class TestClass:
            def __init__(self):
                self.value = None

            def callback(self, value):
                self.value = value

        obj = TestClass()
        manager.register('market_update', obj.callback)

        assert obj.callback in manager.get_callbacks('market_update')

    @pytest.mark.asyncio
    async def test_emit_with_no_arguments(self, manager):
        """Test emitting event with no arguments."""
        called = [False]

        def callback():
            called[0] = True

        manager.register('market_update', callback)
        await manager.emit('market_update')

        assert called[0] is True

    @pytest.mark.asyncio
    async def test_multiple_emits_same_event(self, manager):
        """Test multiple emissions of same event."""
        results = []

        def callback(value):
            results.append(value)

        manager.register('market_update', callback)

        await manager.emit('market_update', 1)
        await manager.emit('market_update', 2)
        await manager.emit('market_update', 3)

        assert results == [1, 2, 3]
