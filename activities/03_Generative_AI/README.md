# Using Google Colab with GitHub Notebooks

This guide provides different methods to open the notebooks in this repository directly in Google Colab.

## Option 1: Open Directly via URL (Fastest)

You can open any notebook in this repository using a specific URL pattern. This requires no code and is the fastest way for public repositories.

**URL Pattern:**
```
https://colab.research.google.com/github/<user>/<repo>/blob/<branch>/<path/to/notebook.ipynb>
```

**Example:**
If the GitHub URL is:
`https://github.com/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb`

The Colab URL will be:
`https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb`

## Option 2: Use the Colab User Interface

1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Select **File** > **Open notebook...** from the menu.
3.  Choose the **GitHub** tab.
4.  Enter the repository URL (e.g., `https://github.com/<user>/<repo>`) and hit Enter.
5.  Colab will list all available notebooks. Click perfectly on the one you want to open.

## Option 3: The `githubtocolab` Shortcut

A quick trick is to change the domain in your address bar:

1.  Navigate to the notebook on GitHub.
2.  Change `github.com` to `githubtocolab.com` in the URL.
3.  Press Enter, and it will redirect you to the notebook in Colab.

## Option 4: Browser Extensions

For one-click access, you can install the "Open in Colab" extension for your browser:
*   [Chrome Web Store](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo)
*   [Firefox Add-ons](https://addons.mozilla.org/en-US/firefox/addon/open-in-colab/)

## Saving Your Work

Once you have opened a notebook in Colab, remember that it is a temporary session. To save your changes:

*   **File** > **Save a copy in Drive** (Create a copy in your personal Google Drive)
*   **File** > **Save a copy to GitHub** (If you have write access or want to save to your own fork)
