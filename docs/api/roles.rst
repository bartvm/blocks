Variable roles
==============

.. autofunction:: blocks.roles.add_role

Roles
-----

All roles are implemented as subclasses of :class:`VariableRole`.

.. autoclass:: blocks.roles.VariableRole

The actual roles are instances of the different subclasses of
:class:`VariableRole`. They are:

.. automodule:: blocks.roles
   :members: INPUT, OUTPUT, AUXILIARY, COST, PARAMETER, WEIGHT, BIAS, FILTER
