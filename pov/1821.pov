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
    sphere { m*<1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 1 }        
    sphere {  m*<1.2741093594903974,1.198582265684995e-18,3.729581819035091>, 1 }
    sphere {  m*<4.985959881404527,5.766749810150201e-18,-0.9335403137814808>, 1 }
    sphere {  m*<-3.836183719470424,8.164965809277259,-2.2886036585540923>, 1}
    sphere { m*<-3.836183719470424,-8.164965809277259,-2.288603658554095>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2741093594903974,1.198582265684995e-18,3.729581819035091>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5 }
    cylinder { m*<4.985959881404527,5.766749810150201e-18,-0.9335403137814808>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5}
    cylinder { m*<-3.836183719470424,8.164965809277259,-2.2886036585540923>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5 }
    cylinder {  m*<-3.836183719470424,-8.164965809277259,-2.288603658554095>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5}

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
    sphere { m*<1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 1 }        
    sphere {  m*<1.2741093594903974,1.198582265684995e-18,3.729581819035091>, 1 }
    sphere {  m*<4.985959881404527,5.766749810150201e-18,-0.9335403137814808>, 1 }
    sphere {  m*<-3.836183719470424,8.164965809277259,-2.2886036585540923>, 1}
    sphere { m*<-3.836183719470424,-8.164965809277259,-2.288603658554095>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2741093594903974,1.198582265684995e-18,3.729581819035091>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5 }
    cylinder { m*<4.985959881404527,5.766749810150201e-18,-0.9335403137814808>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5}
    cylinder { m*<-3.836183719470424,8.164965809277259,-2.2886036585540923>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5 }
    cylinder {  m*<-3.836183719470424,-8.164965809277259,-2.288603658554095>, <1.0818274354141186,-7.062410635429102e-19,0.7357425376360751>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    