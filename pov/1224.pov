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
    sphere { m*<0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 1 }        
    sphere {  m*<0.3558995155591131,-2.7101760725381482e-18,4.0818011774979945>, 1 }
    sphere {  m*<8.214784004462956,4.802452466294345e-18,-1.8401229344151437>, 1 }
    sphere {  m*<-4.448637318129543,8.164965809277259,-2.1831662150319264>, 1}
    sphere { m*<-4.448637318129543,-8.164965809277259,-2.183166215031929>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3558995155591131,-2.7101760725381482e-18,4.0818011774979945>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5 }
    cylinder { m*<8.214784004462956,4.802452466294345e-18,-1.8401229344151437>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5}
    cylinder { m*<-4.448637318129543,8.164965809277259,-2.1831662150319264>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5 }
    cylinder {  m*<-4.448637318129543,-8.164965809277259,-2.183166215031929>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5}

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
    sphere { m*<0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 1 }        
    sphere {  m*<0.3558995155591131,-2.7101760725381482e-18,4.0818011774979945>, 1 }
    sphere {  m*<8.214784004462956,4.802452466294345e-18,-1.8401229344151437>, 1 }
    sphere {  m*<-4.448637318129543,8.164965809277259,-2.1831662150319264>, 1}
    sphere { m*<-4.448637318129543,-8.164965809277259,-2.183166215031929>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3558995155591131,-2.7101760725381482e-18,4.0818011774979945>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5 }
    cylinder { m*<8.214784004462956,4.802452466294345e-18,-1.8401229344151437>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5}
    cylinder { m*<-4.448637318129543,8.164965809277259,-2.1831662150319264>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5 }
    cylinder {  m*<-4.448637318129543,-8.164965809277259,-2.183166215031929>, <0.3128056979654087,-4.061407269518786e-18,1.0821093277230605>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    