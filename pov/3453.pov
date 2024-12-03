#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 1 }        
    sphere {  m*<0.42112780732891036,0.6775950220086309,2.964022707470944>, 1 }
    sphere {  m*<2.9151010965934763,0.6509189192146798,-1.2527415891007911>, 1 }
    sphere {  m*<-1.441222657305671,2.8773588882469063,-0.9974778290655767>, 1}
    sphere { m*<-2.89821248516918,-5.270776783590251,-1.807256481745822>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42112780732891036,0.6775950220086309,2.964022707470944>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5 }
    cylinder { m*<2.9151010965934763,0.6509189192146798,-1.2527415891007911>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5}
    cylinder { m*<-1.441222657305671,2.8773588882469063,-0.9974778290655767>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5 }
    cylinder {  m*<-2.89821248516918,-5.270776783590251,-1.807256481745822>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 1 }        
    sphere {  m*<0.42112780732891036,0.6775950220086309,2.964022707470944>, 1 }
    sphere {  m*<2.9151010965934763,0.6509189192146798,-1.2527415891007911>, 1 }
    sphere {  m*<-1.441222657305671,2.8773588882469063,-0.9974778290655767>, 1}
    sphere { m*<-2.89821248516918,-5.270776783590251,-1.807256481745822>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42112780732891036,0.6775950220086309,2.964022707470944>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5 }
    cylinder { m*<2.9151010965934763,0.6509189192146798,-1.2527415891007911>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5}
    cylinder { m*<-1.441222657305671,2.8773588882469063,-0.9974778290655767>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5 }
    cylinder {  m*<-2.89821248516918,-5.270776783590251,-1.807256481745822>, <0.18039270258721873,0.5488849438283054,-0.02353206364960639>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    