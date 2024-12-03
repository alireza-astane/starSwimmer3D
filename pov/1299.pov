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
    sphere { m*<0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 1 }        
    sphere {  m*<0.47401473985752535,-2.148429867893413e-18,4.041789789020569>, 1 }
    sphere {  m*<7.809948766370067,3.018252731784979e-18,-1.7370425280812667>, 1 }
    sphere {  m*<-4.363667566200382,8.164965809277259,-2.197620750722179>, 1}
    sphere { m*<-4.363667566200382,-8.164965809277259,-2.197620750722182>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.47401473985752535,-2.148429867893413e-18,4.041789789020569>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5 }
    cylinder { m*<7.809948766370067,3.018252731784979e-18,-1.7370425280812667>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5}
    cylinder { m*<-4.363667566200382,8.164965809277259,-2.197620750722179>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5 }
    cylinder {  m*<-4.363667566200382,-8.164965809277259,-2.197620750722182>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5}

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
    sphere { m*<0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 1 }        
    sphere {  m*<0.47401473985752535,-2.148429867893413e-18,4.041789789020569>, 1 }
    sphere {  m*<7.809948766370067,3.018252731784979e-18,-1.7370425280812667>, 1 }
    sphere {  m*<-4.363667566200382,8.164965809277259,-2.197620750722179>, 1}
    sphere { m*<-4.363667566200382,-8.164965809277259,-2.197620750722182>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.47401473985752535,-2.148429867893413e-18,4.041789789020569>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5 }
    cylinder { m*<7.809948766370067,3.018252731784979e-18,-1.7370425280812667>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5}
    cylinder { m*<-4.363667566200382,8.164965809277259,-2.197620750722179>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5 }
    cylinder {  m*<-4.363667566200382,-8.164965809277259,-2.197620750722182>, <0.4150194390566396,-5.688470795947624e-18,1.042367978348273>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    