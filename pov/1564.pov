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
    sphere { m*<0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 1 }        
    sphere {  m*<0.8871317014021568,-9.953672522853504e-19,3.890856636317423>, 1 }
    sphere {  m*<6.377975666265754,4.1306609303038575e-18,-1.3516782818366686>, 1 }
    sphere {  m*<-4.080237365157163,8.164965809277259,-2.2458168988522385>, 1}
    sphere { m*<-4.080237365157163,-8.164965809277259,-2.245816898852242>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8871317014021568,-9.953672522853504e-19,3.890856636317423>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5 }
    cylinder { m*<6.377975666265754,4.1306609303038575e-18,-1.3516782818366686>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5}
    cylinder { m*<-4.080237365157163,8.164965809277259,-2.2458168988522385>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5 }
    cylinder {  m*<-4.080237365157163,-8.164965809277259,-2.245816898852242>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5}

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
    sphere { m*<0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 1 }        
    sphere {  m*<0.8871317014021568,-9.953672522853504e-19,3.890856636317423>, 1 }
    sphere {  m*<6.377975666265754,4.1306609303038575e-18,-1.3516782818366686>, 1 }
    sphere {  m*<-4.080237365157163,8.164965809277259,-2.2458168988522385>, 1}
    sphere { m*<-4.080237365157163,-8.164965809277259,-2.245816898852242>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8871317014021568,-9.953672522853504e-19,3.890856636317423>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5 }
    cylinder { m*<6.377975666265754,4.1306609303038575e-18,-1.3516782818366686>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5}
    cylinder { m*<-4.080237365157163,8.164965809277259,-2.2458168988522385>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5 }
    cylinder {  m*<-4.080237365157163,-8.164965809277259,-2.245816898852242>, <0.7652973241022001,-4.3651236793761385e-18,0.893327157162912>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    