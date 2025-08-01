import pytest

from tempo.core.fast_object_pool import ObjectPool


class Example1:
    def __init__(self, val: int):
        self.val = val

    def __del__(self):
        pass


class Example2:
    def __init__(self, val: int):
        self.val = val

    def __del__(self):
        pass


@pytest.mark.skip
def test_obj_pool():
    x = 1
    pool1 = ObjectPool(
        lambda: Example1(x), lambda obj: Example1(obj.val), max_unused=100
    )

    pool2 = ObjectPool(
        lambda: Example2(x), lambda obj: Example2(obj.val), max_unused=100
    )

    obj1 = pool1.borrow()
    obj1_2 = pool2.borrow()
    x += 1
    obj2 = pool1.borrow()
    obj2_2 = pool2.borrow()
    x += 1
    obj3 = pool1.borrow()
    obj3_2 = pool2.borrow()
    x += 1
    obj4 = pool1.borrow()
    obj4_2 = pool2.borrow()
    x += 1
    obj5 = pool1.borrow()
    obj5_2 = pool2.borrow()
    x += 1

    print(obj1.val)
    print(obj2.val)
    print(obj3.val)

    obj1 = None
    obj1_2 = None

    print(pool1.borrow())
    obj2 = None
    obj2_2 = None

    obj3 = None
    obj3_2 = None
    obj4 = None
    obj4_2 = None

    a = pool1.borrow()
    b = pool1.borrow()
    a = None
    b = None

    obj5 = None
    obj5_2 = None

    for z in pool1.pool:
        assert isinstance(z, Example1)

    for z in pool2.pool:
        assert isinstance(z, Example2)

    import gc

    gc.collect()

    for i in range(10):
        obj = pool1.borrow()
        print("val", obj.val)
        assert isinstance(obj, Example1)
        assert isinstance(obj.val, int)
        assert hasattr(obj, "__pool__")
        assert hasattr(obj, "__original_del__")
        assert hasattr(obj, "__del__")
        obj = None
        gc.collect()
        x += 1
    print(len(pool1.pool))
    print(len(pool2.pool))
