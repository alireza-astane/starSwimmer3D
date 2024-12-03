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
    sphere { m*<-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 1 }        
    sphere {  m*<0.9678681795067569,0.4865568938805991,9.390734997482815>, 1 }
    sphere {  m*<8.335655377829555,0.20146464308833734,-5.179942431591116>, 1 }
    sphere {  m*<-6.560307815859441,6.724546016708977,-3.6891355284095093>, 1}
    sphere { m*<-3.8027383094791163,-7.802161755986194,-2.0105647705141916>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9678681795067569,0.4865568938805991,9.390734997482815>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5 }
    cylinder { m*<8.335655377829555,0.20146464308833734,-5.179942431591116>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5}
    cylinder { m*<-6.560307815859441,6.724546016708977,-3.6891355284095093>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5 }
    cylinder {  m*<-3.8027383094791163,-7.802161755986194,-2.0105647705141916>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5}

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
    sphere { m*<-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 1 }        
    sphere {  m*<0.9678681795067569,0.4865568938805991,9.390734997482815>, 1 }
    sphere {  m*<8.335655377829555,0.20146464308833734,-5.179942431591116>, 1 }
    sphere {  m*<-6.560307815859441,6.724546016708977,-3.6891355284095093>, 1}
    sphere { m*<-3.8027383094791163,-7.802161755986194,-2.0105647705141916>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9678681795067569,0.4865568938805991,9.390734997482815>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5 }
    cylinder { m*<8.335655377829555,0.20146464308833734,-5.179942431591116>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5}
    cylinder { m*<-6.560307815859441,6.724546016708977,-3.6891355284095093>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5 }
    cylinder {  m*<-3.8027383094791163,-7.802161755986194,-2.0105647705141916>, <-0.4512993146934049,-0.5033820199993182,-0.45855509955233176>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    