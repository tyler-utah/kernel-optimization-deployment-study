# Using this artifact as a submodule

This directory is a standalone Git repository. After publishing it to its own
remote, attach it to the talk repository with:

```powershell
git submodule add <experiments-repository-url> experiments
git commit -m "Add empirical studies artifact"
```

Clone the talk and artifact together with:

```powershell
git clone --recurse-submodules <talk-repository-url>
```

Or initialize it after an ordinary clone:

```powershell
git submodule update --init --recursive
```

The talk repository should cite stable artifact commits or release tags rather
than depending on a mutable branch.
