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
    sphere { m*<0.28349793943008694,0.7437906118682385,0.0362064599264702>, 1 }        
    sphere {  m*<0.5242330441717785,0.872500690048564,3.02376123104702>, 1 }
    sphere {  m*<3.0182063334363427,0.8458245872546128,-1.1930030655247128>, 1 }
    sphere {  m*<-1.3381174204628037,3.072264556286841,-0.9377393054894988>, 1}
    sphere { m*<-3.2563113676296047,-5.947711380197313,-2.0147367144427366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5242330441717785,0.872500690048564,3.02376123104702>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5 }
    cylinder { m*<3.0182063334363427,0.8458245872546128,-1.1930030655247128>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5}
    cylinder { m*<-1.3381174204628037,3.072264556286841,-0.9377393054894988>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5 }
    cylinder {  m*<-3.2563113676296047,-5.947711380197313,-2.0147367144427366>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5}

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
    sphere { m*<0.28349793943008694,0.7437906118682385,0.0362064599264702>, 1 }        
    sphere {  m*<0.5242330441717785,0.872500690048564,3.02376123104702>, 1 }
    sphere {  m*<3.0182063334363427,0.8458245872546128,-1.1930030655247128>, 1 }
    sphere {  m*<-1.3381174204628037,3.072264556286841,-0.9377393054894988>, 1}
    sphere { m*<-3.2563113676296047,-5.947711380197313,-2.0147367144427366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5242330441717785,0.872500690048564,3.02376123104702>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5 }
    cylinder { m*<3.0182063334363427,0.8458245872546128,-1.1930030655247128>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5}
    cylinder { m*<-1.3381174204628037,3.072264556286841,-0.9377393054894988>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5 }
    cylinder {  m*<-3.2563113676296047,-5.947711380197313,-2.0147367144427366>, <0.28349793943008694,0.7437906118682385,0.0362064599264702>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    