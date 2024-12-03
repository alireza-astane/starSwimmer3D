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
    sphere { m*<0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 1 }        
    sphere {  m*<0.16472868400541466,-3.444641210407149e-18,4.144121055012986>, 1 }
    sphere {  m*<8.867503466194547,2.6762207071309957e-18,-2.0020960788781057>, 1 }
    sphere {  m*<-4.58952720971436,8.164965809277259,-2.1590851541558784>, 1}
    sphere { m*<-4.58952720971436,-8.164965809277259,-2.159085154155881>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16472868400541466,-3.444641210407149e-18,4.144121055012986>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5 }
    cylinder { m*<8.867503466194547,2.6762207071309957e-18,-2.0020960788781057>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5}
    cylinder { m*<-4.58952720971436,8.164965809277259,-2.1590851541558784>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5 }
    cylinder {  m*<-4.58952720971436,-8.164965809277259,-2.159085154155881>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5}

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
    sphere { m*<0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 1 }        
    sphere {  m*<0.16472868400541466,-3.444641210407149e-18,4.144121055012986>, 1 }
    sphere {  m*<8.867503466194547,2.6762207071309957e-18,-2.0020960788781057>, 1 }
    sphere {  m*<-4.58952720971436,8.164965809277259,-2.1590851541558784>, 1}
    sphere { m*<-4.58952720971436,-8.164965809277259,-2.159085154155881>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16472868400541466,-3.444641210407149e-18,4.144121055012986>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5 }
    cylinder { m*<8.867503466194547,2.6762207071309957e-18,-2.0020960788781057>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5}
    cylinder { m*<-4.58952720971436,8.164965809277259,-2.1590851541558784>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5 }
    cylinder {  m*<-4.58952720971436,-8.164965809277259,-2.159085154155881>, <0.14563673363750224,-4.158280205490281e-18,1.1441812216474956>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    