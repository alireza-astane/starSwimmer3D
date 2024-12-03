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
    sphere { m*<-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 1 }        
    sphere {  m*<0.33587306566516884,0.18435820658572383,5.812528676361793>, 1 }
    sphere {  m*<2.5270818381996007,-0.00419227119968274,-2.161583670689215>, 1 }
    sphere {  m*<-1.8292419156995465,2.2222476978325423,-1.9063199106540019>, 1}
    sphere { m*<-1.5614546946617147,-2.665444244571355,-1.716773625491429>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33587306566516884,0.18435820658572383,5.812528676361793>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5 }
    cylinder { m*<2.5270818381996007,-0.00419227119968274,-2.161583670689215>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5}
    cylinder { m*<-1.8292419156995465,2.2222476978325423,-1.9063199106540019>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5 }
    cylinder {  m*<-1.5614546946617147,-2.665444244571355,-1.716773625491429>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5}

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
    sphere { m*<-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 1 }        
    sphere {  m*<0.33587306566516884,0.18435820658572383,5.812528676361793>, 1 }
    sphere {  m*<2.5270818381996007,-0.00419227119968274,-2.161583670689215>, 1 }
    sphere {  m*<-1.8292419156995465,2.2222476978325423,-1.9063199106540019>, 1}
    sphere { m*<-1.5614546946617147,-2.665444244571355,-1.716773625491429>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33587306566516884,0.18435820658572383,5.812528676361793>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5 }
    cylinder { m*<2.5270818381996007,-0.00419227119968274,-2.161583670689215>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5}
    cylinder { m*<-1.8292419156995465,2.2222476978325423,-1.9063199106540019>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5 }
    cylinder {  m*<-1.5614546946617147,-2.665444244571355,-1.716773625491429>, <-0.20762655580665657,-0.10622624658605698,-0.9323741452380345>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    