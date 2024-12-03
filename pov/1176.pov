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
    sphere { m*<0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 1 }        
    sphere {  m*<0.28014014841191115,-1.4683827716336068e-18,4.106840543391692>, 1 }
    sphere {  m*<8.473776398679895,5.335166681341952e-18,-1.9049760615524776>, 1 }
    sphere {  m*<-4.503986738332687,8.164965809277259,-2.1737252590335485>, 1}
    sphere { m*<-4.503986738332687,-8.164965809277259,-2.173725259033551>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28014014841191115,-1.4683827716336068e-18,4.106840543391692>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5 }
    cylinder { m*<8.473776398679895,5.335166681341952e-18,-1.9049760615524776>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5}
    cylinder { m*<-4.503986738332687,8.164965809277259,-2.1737252590335485>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5 }
    cylinder {  m*<-4.503986738332687,-8.164965809277259,-2.173725259033551>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5}

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
    sphere { m*<0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 1 }        
    sphere {  m*<0.28014014841191115,-1.4683827716336068e-18,4.106840543391692>, 1 }
    sphere {  m*<8.473776398679895,5.335166681341952e-18,-1.9049760615524776>, 1 }
    sphere {  m*<-4.503986738332687,8.164965809277259,-2.1737252590335485>, 1}
    sphere { m*<-4.503986738332687,-8.164965809277259,-2.173725259033551>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28014014841191115,-1.4683827716336068e-18,4.106840543391692>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5 }
    cylinder { m*<8.473776398679895,5.335166681341952e-18,-1.9049760615524776>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5}
    cylinder { m*<-4.503986738332687,8.164965809277259,-2.1737252590335485>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5 }
    cylinder {  m*<-4.503986738332687,-8.164965809277259,-2.173725259033551>, <0.24680649905656976,-2.514234177769533e-18,1.1070246901367302>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    