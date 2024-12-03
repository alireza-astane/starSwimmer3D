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
    sphere { m*<-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 1 }        
    sphere {  m*<0.6204997361686962,-0.2699437493063308,9.22987299632857>, 1 }
    sphere {  m*<7.988286934491507,-0.5550360000985927,-5.340804432745373>, 1 }
    sphere {  m*<-6.9076762591975,5.968045373522065,-3.849997529563767>, 1}
    sphere { m*<-2.0708215573969477,-4.030385240232612,-1.2085356860778083>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6204997361686962,-0.2699437493063308,9.22987299632857>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5 }
    cylinder { m*<7.988286934491507,-0.5550360000985927,-5.340804432745373>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5}
    cylinder { m*<-6.9076762591975,5.968045373522065,-3.849997529563767>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5 }
    cylinder {  m*<-2.0708215573969477,-4.030385240232612,-1.2085356860778083>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5}

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
    sphere { m*<-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 1 }        
    sphere {  m*<0.6204997361686962,-0.2699437493063308,9.22987299632857>, 1 }
    sphere {  m*<7.988286934491507,-0.5550360000985927,-5.340804432745373>, 1 }
    sphere {  m*<-6.9076762591975,5.968045373522065,-3.849997529563767>, 1}
    sphere { m*<-2.0708215573969477,-4.030385240232612,-1.2085356860778083>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6204997361686962,-0.2699437493063308,9.22987299632857>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5 }
    cylinder { m*<7.988286934491507,-0.5550360000985927,-5.340804432745373>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5}
    cylinder { m*<-6.9076762591975,5.968045373522065,-3.849997529563767>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5 }
    cylinder {  m*<-2.0708215573969477,-4.030385240232612,-1.2085356860778083>, <-0.798667758031467,-1.2598826631862492,-0.6194171007065884>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    