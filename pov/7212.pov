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
    sphere { m*<-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 1 }        
    sphere {  m*<0.708474823618349,-0.07835118666436625,9.270613163981327>, 1 }
    sphere {  m*<8.076262021941146,-0.363443437456629,-5.300064265092606>, 1 }
    sphere {  m*<-6.819701171747842,6.159637936164026,-3.809257361911002>, 1}
    sphere { m*<-2.550706426765509,-5.0754807844461265,-1.4307643942139379>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.708474823618349,-0.07835118666436625,9.270613163981327>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5 }
    cylinder { m*<8.076262021941146,-0.363443437456629,-5.300064265092606>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5}
    cylinder { m*<-6.819701171747842,6.159637936164026,-3.809257361911002>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5 }
    cylinder {  m*<-2.550706426765509,-5.0754807844461265,-1.4307643942139379>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5}

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
    sphere { m*<-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 1 }        
    sphere {  m*<0.708474823618349,-0.07835118666436625,9.270613163981327>, 1 }
    sphere {  m*<8.076262021941146,-0.363443437456629,-5.300064265092606>, 1 }
    sphere {  m*<-6.819701171747842,6.159637936164026,-3.809257361911002>, 1}
    sphere { m*<-2.550706426765509,-5.0754807844461265,-1.4307643942139379>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.708474823618349,-0.07835118666436625,9.270613163981327>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5 }
    cylinder { m*<8.076262021941146,-0.363443437456629,-5.300064265092606>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5}
    cylinder { m*<-6.819701171747842,6.159637936164026,-3.809257361911002>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5 }
    cylinder {  m*<-2.550706426765509,-5.0754807844461265,-1.4307643942139379>, <-0.7106926705818133,-1.068290100544284,-0.5786769330538252>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    