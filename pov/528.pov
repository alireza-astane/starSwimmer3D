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
    sphere { m*<-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 1 }        
    sphere {  m*<-1.124931423200621e-18,-4.508426306578239e-18,6.960730993111729>, 1 }
    sphere {  m*<9.428090415820634,-1.173697903298618e-19,-2.669602340221624>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.669602340221624>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.669602340221624>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.124931423200621e-18,-4.508426306578239e-18,6.960730993111729>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5 }
    cylinder { m*<9.428090415820634,-1.173697903298618e-19,-2.669602340221624>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.669602340221624>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.669602340221624>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5}

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
    sphere { m*<-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 1 }        
    sphere {  m*<-1.124931423200621e-18,-4.508426306578239e-18,6.960730993111729>, 1 }
    sphere {  m*<9.428090415820634,-1.173697903298618e-19,-2.669602340221624>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.669602340221624>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.669602340221624>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.124931423200621e-18,-4.508426306578239e-18,6.960730993111729>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5 }
    cylinder { m*<9.428090415820634,-1.173697903298618e-19,-2.669602340221624>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.669602340221624>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.669602340221624>, <-1.1532191202375869e-18,-2.9289186097373817e-18,0.6637309931117095>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    