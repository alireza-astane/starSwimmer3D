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
    sphere { m*<-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 1 }        
    sphere {  m*<0.6860142909628054,-0.12726584119360274,9.260211970956302>, 1 }
    sphere {  m*<8.053801489285602,-0.41235809198586526,-5.310465458117631>, 1 }
    sphere {  m*<-6.8421617044033844,6.110723281634789,-3.8196585549360265>, 1}
    sphere { m*<-2.4326359600064444,-4.818346362026279,-1.3760874311093878>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6860142909628054,-0.12726584119360274,9.260211970956302>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5 }
    cylinder { m*<8.053801489285602,-0.41235809198586526,-5.310465458117631>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5}
    cylinder { m*<-6.8421617044033844,6.110723281634789,-3.8196585549360265>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5 }
    cylinder {  m*<-2.4326359600064444,-4.818346362026279,-1.3760874311093878>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5}

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
    sphere { m*<-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 1 }        
    sphere {  m*<0.6860142909628054,-0.12726584119360274,9.260211970956302>, 1 }
    sphere {  m*<8.053801489285602,-0.41235809198586526,-5.310465458117631>, 1 }
    sphere {  m*<-6.8421617044033844,6.110723281634789,-3.8196585549360265>, 1}
    sphere { m*<-2.4326359600064444,-4.818346362026279,-1.3760874311093878>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6860142909628054,-0.12726584119360274,9.260211970956302>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5 }
    cylinder { m*<8.053801489285602,-0.41235809198586526,-5.310465458117631>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5}
    cylinder { m*<-6.8421617044033844,6.110723281634789,-3.8196585549360265>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5 }
    cylinder {  m*<-2.4326359600064444,-4.818346362026279,-1.3760874311093878>, <-0.733153203237357,-1.1172047550735205,-0.5890781260788491>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    