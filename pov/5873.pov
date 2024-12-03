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
    sphere { m*<-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 1 }        
    sphere {  m*<-0.020853561222380967,0.2790605808269994,8.765698171527275>, 1 }
    sphere {  m*<6.661642444854364,0.09865034446020077,-5.349574488634149>, 1 }
    sphere {  m*<-3.0859342006823787,2.1495166162199477,-2.0096674691181313>, 1}
    sphere { m*<-2.8181469796445473,-2.7381753261839497,-1.8201211839555607>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.020853561222380967,0.2790605808269994,8.765698171527275>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5 }
    cylinder { m*<6.661642444854364,0.09865034446020077,-5.349574488634149>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5}
    cylinder { m*<-3.0859342006823787,2.1495166162199477,-2.0096674691181313>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5 }
    cylinder {  m*<-2.8181469796445473,-2.7381753261839497,-1.8201211839555607>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5}

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
    sphere { m*<-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 1 }        
    sphere {  m*<-0.020853561222380967,0.2790605808269994,8.765698171527275>, 1 }
    sphere {  m*<6.661642444854364,0.09865034446020077,-5.349574488634149>, 1 }
    sphere {  m*<-3.0859342006823787,2.1495166162199477,-2.0096674691181313>, 1}
    sphere { m*<-2.8181469796445473,-2.7381753261839497,-1.8201211839555607>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.020853561222380967,0.2790605808269994,8.765698171527275>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5 }
    cylinder { m*<6.661642444854364,0.09865034446020077,-5.349574488634149>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5}
    cylinder { m*<-3.0859342006823787,2.1495166162199477,-2.0096674691181313>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5 }
    cylinder {  m*<-2.8181469796445473,-2.7381753261839497,-1.8201211839555607>, <-1.4144548496661271,-0.17972333432384674,-1.1260463538520165>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    