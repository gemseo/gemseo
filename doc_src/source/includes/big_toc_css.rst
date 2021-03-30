..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
    File to ..include in a document with a big table of content, to give
    it 'style'

.. raw:: html

  <style type="text/css">
    div.bodywrapper blockquote {
        margin: 0 ;
    }

    div.toctree-wrapper ul {
	margin-top: 0 ;
	margin-bottom: 0 ;
	padding-left: 10px ;
    }

    li.toctree-l1 {
        padding: 0 0 0.5em 0 ;
        list-style-type: none;
        font-size: 150% ;
	font-weight: bold;
        }

    li.toctree-l1 ul {
	padding-left: 40px ;
    }

    li.toctree-l2 {
        font-size: 70% ;
        list-style-type: square;
	font-weight: normal;
        }

    li.toctree-l3 {
        font-size: 85% ;
        list-style-type: circle;
	font-weight: normal;
        }

  </style>
