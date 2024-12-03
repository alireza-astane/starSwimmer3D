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
    sphere { m*<0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 1 }        
    sphere {  m*<0.4308329108729017,0.6959411288365618,2.9696457832197978>, 1 }
    sphere {  m*<2.9248062001374677,0.6692650260426108,-1.247118513351937>, 1 }
    sphere {  m*<-1.43151755376168,2.895704995074837,-0.9918547533167225>, 1}
    sphere { m*<-2.9327324341611214,-5.33603179802984,-1.827257122351912>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4308329108729017,0.6959411288365618,2.9696457832197978>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5 }
    cylinder { m*<2.9248062001374677,0.6692650260426108,-1.247118513351937>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5}
    cylinder { m*<-1.43151755376168,2.895704995074837,-0.9918547533167225>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5 }
    cylinder {  m*<-2.9327324341611214,-5.33603179802984,-1.827257122351912>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5}

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
    sphere { m*<0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 1 }        
    sphere {  m*<0.4308329108729017,0.6959411288365618,2.9696457832197978>, 1 }
    sphere {  m*<2.9248062001374677,0.6692650260426108,-1.247118513351937>, 1 }
    sphere {  m*<-1.43151755376168,2.895704995074837,-0.9918547533167225>, 1}
    sphere { m*<-2.9327324341611214,-5.33603179802984,-1.827257122351912>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4308329108729017,0.6959411288365618,2.9696457832197978>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5 }
    cylinder { m*<2.9248062001374677,0.6692650260426108,-1.247118513351937>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5}
    cylinder { m*<-1.43151755376168,2.895704995074837,-0.9918547533167225>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5 }
    cylinder {  m*<-2.9327324341611214,-5.33603179802984,-1.827257122351912>, <0.19009780613121008,0.5672310506562362,-0.017908987900752266>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    