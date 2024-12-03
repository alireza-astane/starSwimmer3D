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
    sphere { m*<-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 1 }        
    sphere {  m*<0.8241786852543161,0.1736292318955195,9.324194181719093>, 1 }
    sphere {  m*<8.191965883577115,-0.1114630188967427,-5.246483247354839>, 1 }
    sphere {  m*<-6.703997310111873,6.411618354723898,-3.7556763441732333>, 1}
    sphere { m*<-3.129053623462919,-6.335008052042936,-1.6985897802181504>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8241786852543161,0.1736292318955195,9.324194181719093>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5 }
    cylinder { m*<8.191965883577115,-0.1114630188967427,-5.246483247354839>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5}
    cylinder { m*<-6.703997310111873,6.411618354723898,-3.7556763441732333>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5 }
    cylinder {  m*<-3.129053623462919,-6.335008052042936,-1.6985897802181504>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5}

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
    sphere { m*<-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 1 }        
    sphere {  m*<0.8241786852543161,0.1736292318955195,9.324194181719093>, 1 }
    sphere {  m*<8.191965883577115,-0.1114630188967427,-5.246483247354839>, 1 }
    sphere {  m*<-6.703997310111873,6.411618354723898,-3.7556763441732333>, 1}
    sphere { m*<-3.129053623462919,-6.335008052042936,-1.6985897802181504>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8241786852543161,0.1736292318955195,9.324194181719093>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5 }
    cylinder { m*<8.191965883577115,-0.1114630188967427,-5.246483247354839>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5}
    cylinder { m*<-6.703997310111873,6.411618354723898,-3.7556763441732333>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5 }
    cylinder {  m*<-3.129053623462919,-6.335008052042936,-1.6985897802181504>, <-0.5949888089458462,-0.8163096819843982,-0.5250959153160593>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    