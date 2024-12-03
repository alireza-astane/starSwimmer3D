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
    sphere { m*<-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 1 }        
    sphere {  m*<0.8508753446192406,0.2317693432609409,9.336557072280787>, 1 }
    sphere {  m*<8.218662542942043,-0.05332290753132063,-5.234120356793143>, 1 }
    sphere {  m*<-6.677300650746949,6.469758466089316,-3.7433134536115373>, 1}
    sphere { m*<-3.2572278614120527,-6.614146515274464,-1.7579456739625146>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8508753446192406,0.2317693432609409,9.336557072280787>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5 }
    cylinder { m*<8.218662542942043,-0.05332290753132063,-5.234120356793143>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5}
    cylinder { m*<-6.677300650746949,6.469758466089316,-3.7433134536115373>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5 }
    cylinder {  m*<-3.2572278614120527,-6.614146515274464,-1.7579456739625146>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5}

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
    sphere { m*<-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 1 }        
    sphere {  m*<0.8508753446192406,0.2317693432609409,9.336557072280787>, 1 }
    sphere {  m*<8.218662542942043,-0.05332290753132063,-5.234120356793143>, 1 }
    sphere {  m*<-6.677300650746949,6.469758466089316,-3.7433134536115373>, 1}
    sphere { m*<-3.2572278614120527,-6.614146515274464,-1.7579456739625146>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8508753446192406,0.2317693432609409,9.336557072280787>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5 }
    cylinder { m*<8.218662542942043,-0.05332290753132063,-5.234120356793143>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5}
    cylinder { m*<-6.677300650746949,6.469758466089316,-3.7433134536115373>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5 }
    cylinder {  m*<-3.2572278614120527,-6.614146515274464,-1.7579456739625146>, <-0.5682921495809211,-0.7581695706189764,-0.5127330247543628>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    