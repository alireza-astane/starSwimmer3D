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
    sphere { m*<0.28134011508702844,0.7397115543152493,0.03495623011915899>, 1 }        
    sphere {  m*<0.5220752198287201,0.8684216324955747,3.022511001239709>, 1 }
    sphere {  m*<3.0160485090932845,0.8417455297016236,-1.194253295332024>, 1 }
    sphere {  m*<-1.3402752448058624,3.0681854987338517,-0.9389895352968102>, 1}
    sphere { m*<-3.2489923950006783,-5.933875911764404,-2.010496147888275>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5220752198287201,0.8684216324955747,3.022511001239709>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5 }
    cylinder { m*<3.0160485090932845,0.8417455297016236,-1.194253295332024>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5}
    cylinder { m*<-1.3402752448058624,3.0681854987338517,-0.9389895352968102>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5 }
    cylinder {  m*<-3.2489923950006783,-5.933875911764404,-2.010496147888275>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5}

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
    sphere { m*<0.28134011508702844,0.7397115543152493,0.03495623011915899>, 1 }        
    sphere {  m*<0.5220752198287201,0.8684216324955747,3.022511001239709>, 1 }
    sphere {  m*<3.0160485090932845,0.8417455297016236,-1.194253295332024>, 1 }
    sphere {  m*<-1.3402752448058624,3.0681854987338517,-0.9389895352968102>, 1}
    sphere { m*<-3.2489923950006783,-5.933875911764404,-2.010496147888275>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5220752198287201,0.8684216324955747,3.022511001239709>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5 }
    cylinder { m*<3.0160485090932845,0.8417455297016236,-1.194253295332024>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5}
    cylinder { m*<-1.3402752448058624,3.0681854987338517,-0.9389895352968102>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5 }
    cylinder {  m*<-3.2489923950006783,-5.933875911764404,-2.010496147888275>, <0.28134011508702844,0.7397115543152493,0.03495623011915899>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    