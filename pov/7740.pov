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
    sphere { m*<-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 1 }        
    sphere {  m*<0.9683833891101282,0.48767891979804046,9.390973584635454>, 1 }
    sphere {  m*<8.336170587432926,0.20258666900577804,-5.179703844438478>, 1 }
    sphere {  m*<-6.55979260625607,6.725668042626419,-3.6888969412568713>, 1}
    sphere { m*<-3.8050910794556354,-7.8072856295983275,-2.011654308996852>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9683833891101282,0.48767891979804046,9.390973584635454>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5 }
    cylinder { m*<8.336170587432926,0.20258666900577804,-5.179703844438478>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5}
    cylinder { m*<-6.55979260625607,6.725668042626419,-3.6888969412568713>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5 }
    cylinder {  m*<-3.8050910794556354,-7.8072856295983275,-2.011654308996852>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5}

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
    sphere { m*<-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 1 }        
    sphere {  m*<0.9683833891101282,0.48767891979804046,9.390973584635454>, 1 }
    sphere {  m*<8.336170587432926,0.20258666900577804,-5.179703844438478>, 1 }
    sphere {  m*<-6.55979260625607,6.725668042626419,-3.6888969412568713>, 1}
    sphere { m*<-3.8050910794556354,-7.8072856295983275,-2.011654308996852>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9683833891101282,0.48767891979804046,9.390973584635454>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5 }
    cylinder { m*<8.336170587432926,0.20258666900577804,-5.179703844438478>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5}
    cylinder { m*<-6.55979260625607,6.725668042626419,-3.6888969412568713>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5 }
    cylinder {  m*<-3.8050910794556354,-7.8072856295983275,-2.011654308996852>, <-0.4507841050900335,-0.502259994081877,-0.45831651239969406>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    