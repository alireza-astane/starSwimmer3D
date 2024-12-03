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
    sphere { m*<0.5722342586831992,1.0974788133982794,0.2042132489143918>, 1 }        
    sphere {  m*<0.8136971970254777,1.2032967493310072,3.192603080496087>, 1 }
    sphere {  m*<3.3069443860880114,1.2032967493310067,-1.0246791279945275>, 1 }
    sphere {  m*<-1.3693794123780303,3.90581428779537,-0.9437831221135733>, 1}
    sphere { m*<-3.9584610054214946,-7.404968962747322,-2.4739627195808476>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8136971970254777,1.2032967493310072,3.192603080496087>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5 }
    cylinder { m*<3.3069443860880114,1.2032967493310067,-1.0246791279945275>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5}
    cylinder { m*<-1.3693794123780303,3.90581428779537,-0.9437831221135733>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5 }
    cylinder {  m*<-3.9584610054214946,-7.404968962747322,-2.4739627195808476>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5}

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
    sphere { m*<0.5722342586831992,1.0974788133982794,0.2042132489143918>, 1 }        
    sphere {  m*<0.8136971970254777,1.2032967493310072,3.192603080496087>, 1 }
    sphere {  m*<3.3069443860880114,1.2032967493310067,-1.0246791279945275>, 1 }
    sphere {  m*<-1.3693794123780303,3.90581428779537,-0.9437831221135733>, 1}
    sphere { m*<-3.9584610054214946,-7.404968962747322,-2.4739627195808476>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8136971970254777,1.2032967493310072,3.192603080496087>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5 }
    cylinder { m*<3.3069443860880114,1.2032967493310067,-1.0246791279945275>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5}
    cylinder { m*<-1.3693794123780303,3.90581428779537,-0.9437831221135733>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5 }
    cylinder {  m*<-3.9584610054214946,-7.404968962747322,-2.4739627195808476>, <0.5722342586831992,1.0974788133982794,0.2042132489143918>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    