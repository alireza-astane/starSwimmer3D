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
    sphere { m*<0.3357234903802286,0.8425155283553012,0.06646561457506782>, 1 }        
    sphere {  m*<0.5764585951219203,0.9712256065356267,3.0540203856956194>, 1 }
    sphere {  m*<3.070431884386485,0.9445495037416756,-1.1627439108761153>, 1 }
    sphere {  m*<-1.2858918695126613,3.1709894727739036,-0.9074801508409013>, 1}
    sphere { m*<-3.4315289440071135,-6.278935087371783,-2.1162566724767107>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5764585951219203,0.9712256065356267,3.0540203856956194>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5 }
    cylinder { m*<3.070431884386485,0.9445495037416756,-1.1627439108761153>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5}
    cylinder { m*<-1.2858918695126613,3.1709894727739036,-0.9074801508409013>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5 }
    cylinder {  m*<-3.4315289440071135,-6.278935087371783,-2.1162566724767107>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5}

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
    sphere { m*<0.3357234903802286,0.8425155283553012,0.06646561457506782>, 1 }        
    sphere {  m*<0.5764585951219203,0.9712256065356267,3.0540203856956194>, 1 }
    sphere {  m*<3.070431884386485,0.9445495037416756,-1.1627439108761153>, 1 }
    sphere {  m*<-1.2858918695126613,3.1709894727739036,-0.9074801508409013>, 1}
    sphere { m*<-3.4315289440071135,-6.278935087371783,-2.1162566724767107>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5764585951219203,0.9712256065356267,3.0540203856956194>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5 }
    cylinder { m*<3.070431884386485,0.9445495037416756,-1.1627439108761153>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5}
    cylinder { m*<-1.2858918695126613,3.1709894727739036,-0.9074801508409013>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5 }
    cylinder {  m*<-3.4315289440071135,-6.278935087371783,-2.1162566724767107>, <0.3357234903802286,0.8425155283553012,0.06646561457506782>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    