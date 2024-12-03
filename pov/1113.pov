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
    sphere { m*<0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 1 }        
    sphere {  m*<0.1805504963930686,-2.9568983152930693e-18,4.1390692026247>, 1 }
    sphere {  m*<8.813579658939474,3.159206122398253e-18,-1.9888936967051987>, 1 }
    sphere {  m*<-4.577715400105622,8.164965809277259,-2.161110620238712>, 1}
    sphere { m*<-4.577715400105622,-8.164965809277259,-2.1611106202387145>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1805504963930686,-2.9568983152930693e-18,4.1390692026247>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5 }
    cylinder { m*<8.813579658939474,3.159206122398253e-18,-1.9888936967051987>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5}
    cylinder { m*<-4.577715400105622,8.164965809277259,-2.161110620238712>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5 }
    cylinder {  m*<-4.577715400105622,-8.164965809277259,-2.1611106202387145>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5}

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
    sphere { m*<0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 1 }        
    sphere {  m*<0.1805504963930686,-2.9568983152930693e-18,4.1390692026247>, 1 }
    sphere {  m*<8.813579658939474,3.159206122398253e-18,-1.9888936967051987>, 1 }
    sphere {  m*<-4.577715400105622,8.164965809277259,-2.161110620238712>, 1}
    sphere { m*<-4.577715400105622,-8.164965809277259,-2.1611106202387145>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1805504963930686,-2.9568983152930693e-18,4.1390692026247>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5 }
    cylinder { m*<8.813579658939474,3.159206122398253e-18,-1.9888936967051987>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5}
    cylinder { m*<-4.577715400105622,8.164965809277259,-2.161110620238712>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5 }
    cylinder {  m*<-4.577715400105622,-8.164965809277259,-2.1611106202387145>, <0.15954962166610429,-3.7735785586111744e-18,1.1391420644996266>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    