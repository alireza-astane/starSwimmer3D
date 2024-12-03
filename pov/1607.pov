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
    sphere { m*<0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 1 }        
    sphere {  m*<0.9531778951334173,2.5997811671439825e-18,3.8648692612148325>, 1 }
    sphere {  m*<6.1453627623802785,6.017843525369686e-18,-1.2853120237423765>, 1 }
    sphere {  m*<-4.037052688546646,8.164965809277259,-2.253229824407029>, 1}
    sphere { m*<-4.037052688546646,-8.164965809277259,-2.2532298244070326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9531778951334173,2.5997811671439825e-18,3.8648692612148325>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5 }
    cylinder { m*<6.1453627623802785,6.017843525369686e-18,-1.2853120237423765>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5}
    cylinder { m*<-4.037052688546646,8.164965809277259,-2.253229824407029>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5 }
    cylinder {  m*<-4.037052688546646,-8.164965809277259,-2.2532298244070326>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5}

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
    sphere { m*<0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 1 }        
    sphere {  m*<0.9531778951334173,2.5997811671439825e-18,3.8648692612148325>, 1 }
    sphere {  m*<6.1453627623802785,6.017843525369686e-18,-1.2853120237423765>, 1 }
    sphere {  m*<-4.037052688546646,8.164965809277259,-2.253229824407029>, 1}
    sphere { m*<-4.037052688546646,-8.164965809277259,-2.2532298244070326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9531778951334173,2.5997811671439825e-18,3.8648692612148325>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5 }
    cylinder { m*<6.1453627623802785,6.017843525369686e-18,-1.2853120237423765>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5}
    cylinder { m*<-4.037052688546646,8.164965809277259,-2.253229824407029>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5 }
    cylinder {  m*<-4.037052688546646,-8.164965809277259,-2.2532298244070326>, <0.8201608781018793,-2.79808675275166e-18,0.8678147103874191>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    