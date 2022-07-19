This repository contains 5 files: svm.m, diffs.m, PCA_plot.m,
project_report.pdf, and this readme.

The file svm.m is a MATLAB script that reads MIDI files and solves a support
vector machine model that classifies the MIDI files by genre, either classical
music or EDM. The script svm.m calls a function diffs() provided in diffs.m. It
also calls a function readmidi() found in the MATLAB and MIDI library found at
https://kenschutte.com/midi/. The script PCA_plot.m creates plots of  the test
datapoints in the first two principle components and makes it easy to visually
compare the true classifications with the SVM model output. The
project_report.pdf is a report describing the model and the results on a
particular dataset of MIDI files.

To run svm.m, replace the directories in the script with the directories where
your EDM MIDI files, your classical MIDI files, and the MATLAB MIDI library are
located.

This was a project for the class MATH 156 at UCLA. All files were written by me,
John Gardiner.