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
    sphere { m*<-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 1 }        
    sphere {  m*<0.46452761030602846,0.2892537275640438,8.342825702354713>, 1 }
    sphere {  m*<3.2130888101895394,-0.010360891366596986,-3.2912436079805807>, 1 }
    sphere {  m*<-2.06711639138413,2.183720995517481,-2.5660843157789497>, 1}
    sphere { m*<-1.7993291703462981,-2.7039709468864164,-2.376538030616379>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.46452761030602846,0.2892537275640438,8.342825702354713>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5 }
    cylinder { m*<3.2130888101895394,-0.010360891366596986,-3.2912436079805807>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5}
    cylinder { m*<-2.06711639138413,2.183720995517481,-2.5660843157789497>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5 }
    cylinder {  m*<-1.7993291703462981,-2.7039709468864164,-2.376538030616379>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5}

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
    sphere { m*<-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 1 }        
    sphere {  m*<0.46452761030602846,0.2892537275640438,8.342825702354713>, 1 }
    sphere {  m*<3.2130888101895394,-0.010360891366596986,-3.2912436079805807>, 1 }
    sphere {  m*<-2.06711639138413,2.183720995517481,-2.5660843157789497>, 1}
    sphere { m*<-1.7993291703462981,-2.7039709468864164,-2.376538030616379>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.46452761030602846,0.2892537275640438,8.342825702354713>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5 }
    cylinder { m*<3.2130888101895394,-0.010360891366596986,-3.2912436079805807>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5}
    cylinder { m*<-2.06711639138413,2.183720995517481,-2.5660843157789497>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5 }
    cylinder {  m*<-1.7993291703462981,-2.7039709468864164,-2.376538030616379>, <-0.4368094401430606,-0.14485235749529013,-1.6070006854562386>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    