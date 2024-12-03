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
    sphere { m*<-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 1 }        
    sphere {  m*<0.8868813863242814,0.31018347163347637,9.353231021896525>, 1 }
    sphere {  m*<8.254668584647085,0.025091220841215067,-5.217446407177406>, 1 }
    sphere {  m*<-6.641294609041911,6.548172594461856,-3.7266395039957994>, 1}
    sphere { m*<-3.4277163494590885,-6.985437153031196,-1.836896770088192>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8868813863242814,0.31018347163347637,9.353231021896525>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5 }
    cylinder { m*<8.254668584647085,0.025091220841215067,-5.217446407177406>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5}
    cylinder { m*<-6.641294609041911,6.548172594461856,-3.7266395039957994>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5 }
    cylinder {  m*<-3.4277163494590885,-6.985437153031196,-1.836896770088192>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5}

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
    sphere { m*<-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 1 }        
    sphere {  m*<0.8868813863242814,0.31018347163347637,9.353231021896525>, 1 }
    sphere {  m*<8.254668584647085,0.025091220841215067,-5.217446407177406>, 1 }
    sphere {  m*<-6.641294609041911,6.548172594461856,-3.7266395039957994>, 1}
    sphere { m*<-3.4277163494590885,-6.985437153031196,-1.836896770088192>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8868813863242814,0.31018347163347637,9.353231021896525>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5 }
    cylinder { m*<8.254668584647085,0.025091220841215067,-5.217446407177406>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5}
    cylinder { m*<-6.641294609041911,6.548172594461856,-3.7266395039957994>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5 }
    cylinder {  m*<-3.4277163494590885,-6.985437153031196,-1.836896770088192>, <-0.5322861078758804,-0.6797554422464409,-0.49605907513862396>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    