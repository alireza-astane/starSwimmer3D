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
    sphere { m*<1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 1 }        
    sphere {  m*<1.4572728687338858,5.499393992210045e-19,3.6446237908550723>, 1 }
    sphere {  m*<4.284376226484138,6.5848282357653365e-18,-0.6994870037884178>, 1 }
    sphere {  m*<-3.728343428723815,8.164965809277259,-2.3087879973313363>, 1}
    sphere { m*<-3.728343428723815,-8.164965809277259,-2.308787997331339>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4572728687338858,5.499393992210045e-19,3.6446237908550723>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5 }
    cylinder { m*<4.284376226484138,6.5848282357653365e-18,-0.6994870037884178>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5}
    cylinder { m*<-3.728343428723815,8.164965809277259,-2.3087879973313363>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5 }
    cylinder {  m*<-3.728343428723815,-8.164965809277259,-2.308787997331339>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5}

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
    sphere { m*<1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 1 }        
    sphere {  m*<1.4572728687338858,5.499393992210045e-19,3.6446237908550723>, 1 }
    sphere {  m*<4.284376226484138,6.5848282357653365e-18,-0.6994870037884178>, 1 }
    sphere {  m*<-3.728343428723815,8.164965809277259,-2.3087879973313363>, 1}
    sphere { m*<-3.728343428723815,-8.164965809277259,-2.308787997331339>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4572728687338858,5.499393992210045e-19,3.6446237908550723>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5 }
    cylinder { m*<4.284376226484138,6.5848282357653365e-18,-0.6994870037884178>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5}
    cylinder { m*<-3.728343428723815,8.164965809277259,-2.3087879973313363>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5 }
    cylinder {  m*<-3.728343428723815,-8.164965809277259,-2.308787997331339>, <1.2273356593342257,-3.8438966931276135e-19,0.6534390911734999>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    