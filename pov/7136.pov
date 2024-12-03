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
    sphere { m*<-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 1 }        
    sphere {  m*<0.6738112444175565,-0.15384169382894552,9.254560892036801>, 1 }
    sphere {  m*<8.041598442740355,-0.43893394462120794,-5.316116537037131>, 1 }
    sphere {  m*<-6.854364750948636,6.084147428999449,-3.825309633855527>, 1}
    sphere { m*<-2.367403557894664,-4.676282927363148,-1.3458791178758571>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6738112444175565,-0.15384169382894552,9.254560892036801>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5 }
    cylinder { m*<8.041598442740355,-0.43893394462120794,-5.316116537037131>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5}
    cylinder { m*<-6.854364750948636,6.084147428999449,-3.825309633855527>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5 }
    cylinder {  m*<-2.367403557894664,-4.676282927363148,-1.3458791178758571>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5}

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
    sphere { m*<-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 1 }        
    sphere {  m*<0.6738112444175565,-0.15384169382894552,9.254560892036801>, 1 }
    sphere {  m*<8.041598442740355,-0.43893394462120794,-5.316116537037131>, 1 }
    sphere {  m*<-6.854364750948636,6.084147428999449,-3.825309633855527>, 1}
    sphere { m*<-2.367403557894664,-4.676282927363148,-1.3458791178758571>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6738112444175565,-0.15384169382894552,9.254560892036801>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5 }
    cylinder { m*<8.041598442740355,-0.43893394462120794,-5.316116537037131>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5}
    cylinder { m*<-6.854364750948636,6.084147428999449,-3.825309633855527>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5 }
    cylinder {  m*<-2.367403557894664,-4.676282927363148,-1.3458791178758571>, <-0.7453562497826057,-1.143780607708863,-0.5947292049983491>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    