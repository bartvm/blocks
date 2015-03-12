import blocks.roles
from six.moves import cPickle


def test_role_serialization():
    """Test that roles compare equal before and after serialization."""
    roles = [blocks.roles.INPUT,
             blocks.roles.OUTPUT,
             blocks.roles.COST,
             blocks.roles.PARAMETER,
             blocks.roles.AUXILIARY,
             blocks.roles.WEIGHT,
             blocks.roles.BIAS,
             blocks.roles.FILTER]

    for role in roles:
        deserialized = cPickle.loads(cPickle.dumps(role))
        assert deserialized == role
