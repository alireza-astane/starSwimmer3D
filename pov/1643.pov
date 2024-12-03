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
    sphere { m*<0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 1 }        
    sphere {  m*<1.0081573144377813,7.993318379460205e-19,3.8427873917505293>, 1 }
    sphere {  m*<5.950547609651796,4.267527353252372e-18,-1.2287545262346813>, 1 }
    sphere {  m*<-4.001576065950789,8.164965809277259,-2.2593537732048743>, 1}
    sphere { m*<-4.001576065950789,-8.164965809277259,-2.259353773204878>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0081573144377813,7.993318379460205e-19,3.8427873917505293>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5 }
    cylinder { m*<5.950547609651796,4.267527353252372e-18,-1.2287545262346813>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5}
    cylinder { m*<-4.001576065950789,8.164965809277259,-2.2593537732048743>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5 }
    cylinder {  m*<-4.001576065950789,-8.164965809277259,-2.259353773204878>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5}

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
    sphere { m*<0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 1 }        
    sphere {  m*<1.0081573144377813,7.993318379460205e-19,3.8427873917505293>, 1 }
    sphere {  m*<5.950547609651796,4.267527353252372e-18,-1.2287545262346813>, 1 }
    sphere {  m*<-4.001576065950789,8.164965809277259,-2.2593537732048743>, 1}
    sphere { m*<-4.001576065950789,-8.164965809277259,-2.259353773204878>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0081573144377813,7.993318379460205e-19,3.8427873917505293>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5 }
    cylinder { m*<5.950547609651796,4.267527353252372e-18,-1.2287545262346813>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5}
    cylinder { m*<-4.001576065950789,8.164965809277259,-2.2593537732048743>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5 }
    cylinder {  m*<-4.001576065950789,-8.164965809277259,-2.259353773204878>, <0.8655750746928839,-3.460429692207518e-18,0.8461722414028314>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    