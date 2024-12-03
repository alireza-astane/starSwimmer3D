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
    sphere { m*<-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 1 }        
    sphere {  m*<0.8444233025175029,0.21771805537500089,9.333569211915453>, 1 }
    sphere {  m*<8.212210500840307,-0.06737419541726064,-5.237108217158477>, 1 }
    sphere {  m*<-6.683752692848685,6.455707178203373,-3.74630131397687>, 1}
    sphere { m*<-3.2263961689752207,-6.547001109275953,-1.7436679010689804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8444233025175029,0.21771805537500089,9.333569211915453>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5 }
    cylinder { m*<8.212210500840307,-0.06737419541726064,-5.237108217158477>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5}
    cylinder { m*<-6.683752692848685,6.455707178203373,-3.74630131397687>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5 }
    cylinder {  m*<-3.2263961689752207,-6.547001109275953,-1.7436679010689804>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5}

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
    sphere { m*<-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 1 }        
    sphere {  m*<0.8444233025175029,0.21771805537500089,9.333569211915453>, 1 }
    sphere {  m*<8.212210500840307,-0.06737419541726064,-5.237108217158477>, 1 }
    sphere {  m*<-6.683752692848685,6.455707178203373,-3.74630131397687>, 1}
    sphere { m*<-3.2263961689752207,-6.547001109275953,-1.7436679010689804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8444233025175029,0.21771805537500089,9.333569211915453>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5 }
    cylinder { m*<8.212210500840307,-0.06737419541726064,-5.237108217158477>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5}
    cylinder { m*<-6.683752692848685,6.455707178203373,-3.74630131397687>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5 }
    cylinder {  m*<-3.2263961689752207,-6.547001109275953,-1.7436679010689804>, <-0.5747441916826591,-0.7722208585049165,-0.5157208851196954>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    