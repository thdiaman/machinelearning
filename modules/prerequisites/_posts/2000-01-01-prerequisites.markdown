---
title: Prerequisites
---

# Prerequisites
Starting with Machine Learning and scikit-learn and understanding this tutorial is simple.

<div>
After checking that you meet all the prerequisites, you can
<a href="{{site.baseurl}}/modules/tutorial/introduction/">start the tutorial</a>.
</div>

## Software Prerequisites
Typically, you should be able to run this tutorial on any python 3 distribution that
includes numpy, scipy, pandas, matplotlib and scikit-learn.
The pydotplus library is also required if you want to be able to visualize the decision
trees created using scikit-learn.
However, you are strongly advised to use one of the following two pre-bundled python
distributions that include most required python packages.

### Option 1: Use Anaconda
[Anaconda](https://anaconda.org/anaconda) has most of the required packages
and is available for Windows, Linux, or Mac OS.
To install Anaconda on your OS, you have to download the respective installer
(Anaconda version 5.2, python version 3.6, either 64-bit or 32-bit according to your OS) from its
[downloads page](https://www.anaconda.com/download/).

After installing Anaconda, you can also open the Anaconda Prompt and install pydotplus by
typing in:

```
conda install pydotplus
```

AFter confirming the changes that are recommended, pydotplus will be installed.

### Option 2: Use WinPython (Windows only)
[WinPython](http://winpython.sourceforge.net/) is another option, which, however, only
works for Windows. If you want to use WinPython you can download the version 3.6.3.0
of WinPython with QT5, which is available
[here](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.3.0/WinPython-64bit-3.6.3.0Qt5.exe) for 64-bit OSes
and [here](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.3.0/WinPython-32bit-3.6.3.0Qt5.exe/download) for
32-bit OSes.

After installing WinPython, you can also open the WinPython Command Prompt and install pydotplus by
typing in:

```
pip install pydotplus
```

## Other Prerequisites
In terms of other prerequisites etc., you are advised to have the following:

- some basic understanding of data manipulation in Python (numpy, pandas, matplotlib)
- a nice attitude towards learning, because it's going to be fun!

## IDE Prerequisites
As an IDE, you can use anything you want. I use [Spyder](https://pythonhosted.org/spyder/)
for this, which is also preinstalled in the distributions mentioned above.
Having said this, to correctly run all the exampless in Spyder (with IPython), you may
have to select Tools --> Preferences and inside the IPython console and the Graphics tab
set Backend to Automatic. This is shown in the following screenshot:

![Spyder Option for Graphics Backend]({{site.baseurl}}/img/spyder.png)

<div>
Having checked that you meet all the prerequisites, you can
<a href="{{site.baseurl}}/modules/tutorial/introduction/">start the tutorial</a>.
</div>
