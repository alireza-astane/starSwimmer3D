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
    sphere { m*<-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 1 }        
    sphere {  m*<0.49527312785227307,0.26958214801323166,7.790704716848182>, 1 }
    sphere {  m*<2.479770690448506,-0.029487383569680517,-2.7487213270049975>, 1 }
    sphere {  m*<-1.8765530634506409,2.1969525854625447,-2.4934575669697843>, 1}
    sphere { m*<-1.608765842412809,-2.6907393569413527,-2.303911281807211>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49527312785227307,0.26958214801323166,7.790704716848182>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5 }
    cylinder { m*<2.479770690448506,-0.029487383569680517,-2.7487213270049975>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5}
    cylinder { m*<-1.8765530634506409,2.1969525854625447,-2.4934575669697843>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5 }
    cylinder {  m*<-1.608765842412809,-2.6907393569413527,-2.303911281807211>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5}

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
    sphere { m*<-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 1 }        
    sphere {  m*<0.49527312785227307,0.26958214801323166,7.790704716848182>, 1 }
    sphere {  m*<2.479770690448506,-0.029487383569680517,-2.7487213270049975>, 1 }
    sphere {  m*<-1.8765530634506409,2.1969525854625447,-2.4934575669697843>, 1}
    sphere { m*<-1.608765842412809,-2.6907393569413527,-2.303911281807211>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49527312785227307,0.26958214801323166,7.790704716848182>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5 }
    cylinder { m*<2.479770690448506,-0.029487383569680517,-2.7487213270049975>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5}
    cylinder { m*<-1.8765530634506409,2.1969525854625447,-2.4934575669697843>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5 }
    cylinder {  m*<-1.608765842412809,-2.6907393569413527,-2.303911281807211>, <-0.2549377035577508,-0.1315213589560547,-1.5195118015538163>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    