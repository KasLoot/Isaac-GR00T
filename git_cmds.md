# Managing a Forked GitHub Repository

This guide explains the common workflow for managing a forked repository on GitHub, including pushing and pulling changes from both your fork and the original repository.

In these examples, we assume:
* `origin` refers to **your forked repository** on your GitHub account.
* `upstream` refers to the **original repository** you forked from.
* You are working on a branch, for example, `main` or `feature-branch`.

***

### 1. Push Updates to Your Fork

After you've committed your changes locally (`git commit`), you can push them to your own forked repository on GitHub.

```bash
# Push your committed changes to your 'origin' remote (your fork)
git push origin <branch-name>

# For example, to push the 'main' branch:
git push origin main
```

***

### 2. Push Updates to the Original Repo (Upstream)

You typically can't push directly to the original repository unless you are a collaborator. The standard way to contribute is by creating a **Pull Request (PR)**.

1.  **Push to your fork first**: Make sure your changes are pushed to your forked repository (as shown in step 1).
2.  **Create a Pull Request**:
    * Go to your forked repository's page on GitHub.
    * You will often see a banner that says "This branch is X commits ahead of the original-repo:main." with a "Contribute" button. Click **Open pull request**.
    * If you don't see the banner, click the **Pull requests** tab and then the **New pull request** button.
    * Review the changes and submit the pull request. The maintainers of the original repository can then review your changes and merge them.

***

### 3. Pull Your Updates to a Local Machine

If you've made changes on your GitHub fork directly or are setting up your project on a new machine, you can pull those changes down.

* **For a new setup**: Clone your forked repository.
    ```bash
    # Get the URL from the "Code" button on your forked repo's GitHub page
    git clone <your-fork-url>
    ```
* **For an existing local copy**: If you already have the repository cloned, just pull the latest changes from your fork.
    ```bash
    # Pull changes from your 'origin' remote (your fork)
    git pull origin <branch-name>
    ```

***

### 4. Pull Updates from the Original Repo and Merge

To keep your fork synchronized with the latest changes from the original (upstream) project, you need to configure a remote for it and then pull and merge its changes.

1.  **Add the original repo as a remote (only need to do this once)**:
    ```bash
    # Get the URL from the original repository's GitHub page
    git remote add upstream <original-repo-url>

    # Verify that it was added
    git remote -v
    # You should now see both 'origin' and 'upstream' listed
    ```

2.  **Fetch the latest updates from the original repo**:
    ```bash
    # This downloads the latest changes from the 'upstream' remote
    git fetch upstream
    ```

3.  **Merge the updates into your local branch**:
    * First, make sure you are on the branch you want to update (e.g., `main`).
        ```bash
        git checkout main
        ```
    * Then, merge the changes from the original repo's corresponding branch.
        ```bash
        # This merges the 'main' branch from 'upstream' into your current local 'main' branch
        git merge upstream/main
        ```
    * If there are any **merge conflicts**, Git will prompt you to resolve them before you can finalize the merge.

4.  **(Optional) Push the merged updates to your fork**: After merging, your local branch is now up-to-date. You can push this updated version to your fork on GitHub to keep it in sync.
    ```bash
    git push origin main
    ```