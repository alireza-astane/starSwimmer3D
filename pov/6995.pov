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
    sphere { m*<-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 1 }        
    sphere {  m*<0.6139490708214665,-0.37374122690311096,9.233526744467259>, 1 }
    sphere {  m*<7.96930050882144,-0.4626615028974669,-5.345966545578074>, 1 }
    sphere {  m*<-6.9466731305949905,5.935056352905287,-3.7650096950115937>, 1}
    sphere { m*<-2.022078185760922,-3.947221405506322,-1.2200544993725164>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6139490708214665,-0.37374122690311096,9.233526744467259>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5 }
    cylinder { m*<7.96930050882144,-0.4626615028974669,-5.345966545578074>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5}
    cylinder { m*<-6.9466731305949905,5.935056352905287,-3.7650096950115937>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5 }
    cylinder {  m*<-2.022078185760922,-3.947221405506322,-1.2200544993725164>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5}

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
    sphere { m*<-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 1 }        
    sphere {  m*<0.6139490708214665,-0.37374122690311096,9.233526744467259>, 1 }
    sphere {  m*<7.96930050882144,-0.4626615028974669,-5.345966545578074>, 1 }
    sphere {  m*<-6.9466731305949905,5.935056352905287,-3.7650096950115937>, 1}
    sphere { m*<-2.022078185760922,-3.947221405506322,-1.2200544993725164>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6139490708214665,-0.37374122690311096,9.233526744467259>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5 }
    cylinder { m*<7.96930050882144,-0.4626615028974669,-5.345966545578074>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5}
    cylinder { m*<-6.9466731305949905,5.935056352905287,-3.7650096950115937>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5 }
    cylinder {  m*<-2.022078185760922,-3.947221405506322,-1.2200544993725164>, <-0.8086732239263426,-1.269350262639889,-0.6242296808496192>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    