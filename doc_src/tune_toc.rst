..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. raw:: html

   <SCRIPT>
   //Function to make the index toctree collapsible
   $(function () {
       $('.toctree-l2')
           .click(function(event){
               if (event.target.tagName.toLowerCase() != "a") {
                   if ($(this).children('ul').length > 0) {
                        $(this).attr('data-content',
                            (!$(this).children('ul').is(':hidden')) ? '\uf0fe' : '\uf146');
                       $(this).children('ul').toggle();
                   }
                   return true; //Makes links clickable
               }
           })
           .mousedown(function(event){ return false; }) //Firefox highlighting fix
           .children('ul').hide();
       // Initialize the values
       $('li.toctree-l2:not(:has(ul))').attr('data-content', '\uf0c8');
       $('li.toctree-l2:has(ul)').attr('data-content', '\uf0fe');
       $('li.toctree-l2:has(ul)').css('cursor', 'pointer');

       $('.toctree-l2').hover(
           function () {
               if ($(this).children('ul').length > 0) {
                   $(this).css('background-color', '#D0D0D0').children('ul').css('background-color', '#F0F0F0');
                   $(this).attr('data-content',
                       (!$(this).children('ul').is(':hidden')) ? '\uf146' : '\uf0fe');
               }
               else {
                   $(this).css('background-color', '#F9F9F9');
               }
           },
           function () {
               $(this).css('background-color', 'white').children('ul').css('background-color', 'white');
               if ($(this).children('ul').length > 0) {
                   $(this).attr('data-content',
                       (!$(this).children('ul').is(':hidden')) ? '\uf146' : '\uf0fe');
               }
           }
       );
   });

   </SCRIPT>

  <style type="text/css">
    li.toctree-l2 {
        padding: 0.25em 0 0.25em 0 ;
        list-style-type: none;
    }

    li.toctree-l3  {
       list-style-type: square;

    }

    li.toctree-l2 ul {
        padding-left: 40px ;
    }


    li.toctree-l2:before {
        font-family: "Font Awesome 5 Free";
        content: attr(data-content) ;
        display: inline-block;
        width: 20px;
    }

  </style>
