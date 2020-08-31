# Sparse OpticalFlow
A simple introduction to optical flow and an implementation on a dataset.

There are two main types of optical flow - Sparse and Dense with the major difference between the two
being that sparse optical flow identifies certain features (usually edges) whose motion is to be tracked whereas dense optical flow gives flow vectors of an entire frame.

In layman's terms, optical flow is tracking the motion of objects between consecutive frames of a sequence. 
This "motion" is the relative motion between the camera and the object.

The fundamental assumptions are:

1) That the pixel intensities (of an object) remain constant between frames.

2) Neighbouring pixels have similar motion.

The motion is considered to be small and thus optical flow works better in tracking slow moving objects.

The code has comments for further understanding.
