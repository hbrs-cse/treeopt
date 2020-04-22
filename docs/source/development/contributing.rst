.. _contributing:

.. toctree::
    :glob:

******************
Contribution Guide
******************

Begin by reading the section :doc:`getting_started`.
Make sure you are on your master branch and your local copy is up-to-date:

.. code-block:: bash

    git checkout master
    git pull

Create a new feature branch
---------------------------

If you would like to contribute, begin by creating a feature branch

.. code-block:: bash

   git checkout -b jk/issue-42-my-awesome-new-feature

Ideally, the name of the branch should have a speaking name.
It is good practice to prepend a private branch with your initials.
If you are writing a feature that is referenced in an issue on Github, include the issue number in your branch name.

Now implement your modifications. 
The source code of the library is contained in the subdirectory ``treeopt``.
Any new feature should be thoroughly tested, so please add unit tests in the ``tests`` subdirectory.

Commit your changes
-------------------

.. code-block:: bash

   git commit -m 'implemented an awesome new feature
   fixes #42'

The commit message should be a short, but good description of the changes. 
Make as many commits as you need and try to change only one thing.
A good rule of thumb is, that using the word "and" in a commit message is a sign, 
that you could have used two commits instead.

You can add a reference to an Github issue number using a hashtag.

Push your local branch
----------------------

.. code-block:: bash
 
   git push -u origin HEAD

This will generate a new branch in the remote repository with the same name as your local branch
and the sync the two branches.
After the remote branch has been created, the command

.. code-block:: bash
   
   git push

suffices to push your local changes to the remote branch.

Create a Pull Request
---------------------

If you are satisfied with your changes, navigate to the repo on Github and open a Pull Request.
You can prepend the Pull Request name with `[WIP]` to label it as "Work in Progress".
This way, noone will prematurally review your unfinished work.