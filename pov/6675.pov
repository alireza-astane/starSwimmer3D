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
    sphere { m*<-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 1 }        
    sphere {  m*<0.3434176090390948,-0.14923522032590417,9.0956658021254>, 1 }
    sphere {  m*<7.698769047039067,-0.23815549632026045,-5.483827487919937>, 1 }
    sphere {  m*<-5.746119277405111,4.774448237795646,-3.152211786620103>, 1}
    sphere { m*<-2.3576293705026115,-3.5722154130778283,-1.3916390017722686>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3434176090390948,-0.14923522032590417,9.0956658021254>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5 }
    cylinder { m*<7.698769047039067,-0.23815549632026045,-5.483827487919937>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5}
    cylinder { m*<-5.746119277405111,4.774448237795646,-3.152211786620103>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5 }
    cylinder {  m*<-2.3576293705026115,-3.5722154130778283,-1.3916390017722686>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5}

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
    sphere { m*<-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 1 }        
    sphere {  m*<0.3434176090390948,-0.14923522032590417,9.0956658021254>, 1 }
    sphere {  m*<7.698769047039067,-0.23815549632026045,-5.483827487919937>, 1 }
    sphere {  m*<-5.746119277405111,4.774448237795646,-3.152211786620103>, 1}
    sphere { m*<-2.3576293705026115,-3.5722154130778283,-1.3916390017722686>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3434176090390948,-0.14923522032590417,9.0956658021254>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5 }
    cylinder { m*<7.698769047039067,-0.23815549632026045,-5.483827487919937>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5}
    cylinder { m*<-5.746119277405111,4.774448237795646,-3.152211786620103>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5 }
    cylinder {  m*<-2.3576293705026115,-3.5722154130778283,-1.3916390017722686>, <-1.0938786264124087,-0.9236064669305074,-0.7702348229605875>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    